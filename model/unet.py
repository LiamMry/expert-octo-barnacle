# Adapted from: https://github.com/DPS2022/diffusion-posterior-sampling
# Rewritten for robustness with 1D data, small 2D images, and arbitrary sizes.

from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import functools

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import checkpoint, conv_nd, linear, avg_pool_nd, zero_module, timestep_embedding

NUM_CLASSES = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_num_groups(channels: int, preferred: int = 32) -> int:
    """
    Return the largest divisor of `channels` that is <= `preferred`.
    Falls back to 1 (equivalent to LayerNorm per-channel) if needed.
    """
    for g in [preferred, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1  # always valid


def get_num_heads(channels: int, num_head_channels: int) -> int:
    """
    Return the number of attention heads such that each head has
    `num_head_channels` channels, clamped so we never exceed `channels`.
    Falls back to 1 head if channels < num_head_channels.
    """
    if channels < num_head_channels:
        return 1
    # Find largest divisor of channels <= num_head_channels
    for hc in [num_head_channels, num_head_channels // 2, num_head_channels // 4, 8, 4, 2, 1]:
        if hc > 0 and channels % hc == 0:
            return channels // hc
    return 1


def safe_normalization(channels: int):
    """GroupNorm with an automatically chosen number of groups."""
    num_groups = get_num_groups(channels)
    return GroupNorm32(num_groups, channels)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that casts to float32 for numerical stability."""
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def auto_channel_mult(input_size: int, min_spatial: int = 4) -> tuple:
    """
    Automatically generate a `channel_mult` tuple for an arbitrary input_size.
    Downsamples until spatial size would fall below `min_spatial`.
    """
    levels = 0
    s = input_size
    while s // 2 >= min_spatial:
        if (input_size % (2 ** (levels + 1))) != 0:
            break                        # ← stop before creating a bad divisor
        s //= 2
        levels += 1
    levels = max(levels, 1)
    return tuple(min(2 ** i, 8) for i in range(levels))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(
    dims,               # NEW: 1 for 1-D signals, 2 for images (default), 3 for volumes
    image_size,
    in_channels,        # NEW: explicit, no more guessing
    out_channels,       # NEW: if None, defaults to in_channels
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    model_path='',
):
    # --- channel_mult ---
    if channel_mult == "":
        channel_mult = auto_channel_mult(image_size)
    elif isinstance(channel_mult, str):
        channel_mult = tuple(int(c) for c in channel_mult.split(","))

    # --- attention_resolutions ---
    attention_ds = []
    if isinstance(attention_resolutions, int):
        attention_ds.append(image_size // attention_resolutions)
    elif isinstance(attention_resolutions, str):
        for res in attention_resolutions.split(","):
            r = int(res)
            ds = image_size // r
            # Only add if the downsampled size is valid (>= 1)
            if ds >= 1:
                attention_ds.append(ds)
    else:
        raise NotImplementedError

    # --- divisibility guard ---
    num_ds = len(channel_mult) - 1          # number of downsampling ops
    required_divisor = 2 ** num_ds
    if image_size % required_divisor != 0:
        raise ValueError(
            f"image_size={image_size} must be divisible by 2^(len(channel_mult)-1)="
            f"{required_divisor} for the chosen channel_mult={channel_mult}. "
            f"Either pad your data or adjust channel_mult."
        )

    model = UNetModel(
        image_size=image_size,
        in_channels=in_channels,  # sensible defaults; override via subclass
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        dims=dims,
    )

    if model_path:
        try:
            model.load_state_dict(th.load(model_path, map_location='cpu'))
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Could not load weights ({e}). Using random initialization.")

    return model


# ---------------------------------------------------------------------------
# Pooling / positional helpers
# ---------------------------------------------------------------------------

class AttentionPool2d(nn.Module):
    """Adapted from CLIP."""
    def __init__(self, spacial_dim, embed_dim, num_heads_channels, output_dim=None):
        super().__init__()
        num_tokens = spacial_dim ** 2 if dims == 2 else spacial_dim
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, num_tokens + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = max(1, embed_dim // num_heads_channels)
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    Upsampling layer with optional convolution.
    Works for dims=1, 2, or 3.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        elif self.dims == 1:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Downsampling layer with optional convolution.
    Works for dims=1, 2, or 3.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    Residual block conditioned on timestep embeddings.
    Robustly handles any channel count via safe_normalization().
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            safe_normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            safe_normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    Self-attention block that gracefully adapts to any channel count:
    - Automatically finds a valid number of heads.
    - Uses safe_normalization() for GroupNorm compatibility.
    - Can be disabled entirely (replaced by Identity) when channels are too small.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        min_channels_for_attention=8,   # NEW: skip attention below this threshold
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        self.disabled = channels < min_channels_for_attention

        if self.disabled:
            # Too few channels — attention is meaningless; use identity
            return

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            # Robustly find a valid number of heads
            self.num_heads = get_num_heads(channels, num_head_channels)

        # Clamp to at least 1 head
        self.num_heads = max(1, self.num_heads)

        self.norm = safe_normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = (
            QKVAttention(self.num_heads)
            if use_new_attention_order
            else QKVAttentionLegacy(self.num_heads)
        )
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        if self.disabled:
            return x
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


# ---------------------------------------------------------------------------
# Attention implementations
# ---------------------------------------------------------------------------

def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """Split heads before split qkv (legacy order)."""
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """Split qkv before split heads (new order)."""
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNetModel(nn.Module):
    """
    Full UNet with attention and timestep embedding.

    New vs original:
    - `dims` parameter: 1 (signals), 2 (images, default), 3 (volumes).
    - GroupNorm groups auto-selected via safe_normalization().
    - Attention heads auto-selected via get_num_heads().
    - Attention silently disabled on very narrow feature maps.
    - auto_channel_mult() removes the hardcoded size restriction.
    """
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        min_channels_for_attention=8,   # attention disabled below this
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dims = dims
        self.min_channels_for_attention = min_channels_for_attention

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(self._make_attention(ch, num_heads, num_head_channels, use_new_attention_order, use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                                 use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True)
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            self._make_attention(ch, num_heads, num_head_channels, use_new_attention_order, use_checkpoint),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich, time_embed_dim, dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(self._make_attention(ch, num_heads_upsample, num_head_channels, use_new_attention_order, use_checkpoint))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                                 use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True)
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            safe_normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def _make_attention(self, ch, n_heads, n_head_channels, new_order, use_checkpoint):
        """Factory that always returns a valid AttentionBlock."""
        return AttentionBlock(
            ch,
            num_heads=n_heads,
            num_head_channels=n_head_channels,
            use_checkpoint=use_checkpoint,
            use_new_attention_order=new_order,
            min_channels_for_attention=self.min_channels_for_attention,
        )

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        :param x:          [N x C x ...] input tensor (1D, 2D, or 3D)
        :param timesteps:  [N] timestep indices
        :param y:          [N] class labels (only if class-conditional)
        """
        assert (y is not None) == (self.num_classes is not None), \
            "must specify y if and only if the model is class-conditional"

        hs = []
        # timesteps are continuous t ∈ [0, T=1]; the sinusoidal embedding was
        # designed for integer steps 0..1000, so rescale (score_sde convention)
        # to spread t across the embedding's usable frequency range.
        emb = self.time_embed(timestep_embedding(timesteps * 999.0, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


# ---------------------------------------------------------------------------
# Specialised variants
# ---------------------------------------------------------------------------

# class SuperResModel(UNetModel):
#     """UNet for super-resolution; conditions on a low-res input."""
#     def __init__(self, image_size, in_channels, *args, **kwargs):
#         super().__init__(image_size, in_channels * 2, *args, **kwargs)

#     def forward(self, x, timesteps, low_res=None, **kwargs):
#         *_, new_h, new_w = x.shape
#         mode = "bilinear" if self.dims == 2 else "linear"
#         upsampled = F.interpolate(low_res, size=x.shape[2:], mode=mode, align_corners=False)
#         x = th.cat([x, upsampled], dim=1)
#         return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    Encoder-only (half) UNet that maps input → latent vector.
    Supports dims=1/2/3 and arbitrary sizes.
    """
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
        min_channels_for_attention=8,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.dims = dims
        self.min_channels_for_attention = min_channels_for_attention

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout, out_channels=int(mult * model_channels),
                             dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(
                        ch, num_heads=num_heads, num_head_channels=num_head_channels,
                        use_checkpoint=use_checkpoint, use_new_attention_order=use_new_attention_order,
                        min_channels_for_attention=min_channels_for_attention,
                    ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                                 use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True)
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels,
                           use_checkpoint=use_checkpoint, use_new_attention_order=use_new_attention_order,
                           min_channels_for_attention=min_channels_for_attention),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        )
        self._feature_size += ch
        self.pool = pool

        if pool == "adaptive":
            self.out = nn.Sequential(
                safe_normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)) if dims == 2 else nn.AdaptiveAvgPool1d(1),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                safe_normalization(ch),
                nn.SiLU(),
                AttentionPool2d((image_size // ds), ch, num_head_channels, out_channels),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                safe_normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected pool={pool}")

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        # Continuous t ∈ [0, T=1] rescaled to 0..999 (same convention as UNetModel).
        emb = self.time_embed(timestep_embedding(timesteps * 999.0, self.model_channels))
        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                # mean over all spatial dims
                results.append(h.type(x.dtype).mean(dim=list(range(2, h.ndim))))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=list(range(2, h.ndim))))
            h = th.cat(results, dim=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)


# ---------------------------------------------------------------------------
# GAN components (unchanged except normalization)
# ---------------------------------------------------------------------------

# class NLayerDiscriminator(nn.Module):
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
#         super().__init__()
#         use_bias = (
#             norm_layer.func == nn.InstanceNorm2d
#             if isinstance(norm_layer, functools.partial)
#             else norm_layer == nn.InstanceNorm2d
#         )
#         kw, padw = 4, 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         for n in range(1, n_layers):
#             nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True),
#             ]
#         nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=2, padding=padw),
#             nn.Dropout(0.5),
#         ]
#         if use_sigmoid:
#             sequence.append(nn.Sigmoid())
#         self.model = nn.Sequential(*sequence)

#     def forward(self, input):
#         return self.model(input)


# class GANLoss(nn.Module):
#     def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
#         super().__init__()
#         self.register_buffer('real_label', th.tensor(target_real_label))
#         self.register_buffer('fake_label', th.tensor(target_fake_label))
#         self.gan_mode = gan_mode
#         if gan_mode == 'lsgan':
#             self.loss = nn.MSELoss()
#         elif gan_mode == 'vanilla':
#             self.loss = nn.BCEWithLogitsLoss()
#         elif gan_mode == 'wgangp':
#             self.loss = None
#         else:
#             raise NotImplementedError(f'gan mode {gan_mode} not implemented')

#     def get_target_tensor(self, prediction, target_is_real):
#         target = self.real_label if target_is_real else self.fake_label
#         return target.expand_as(prediction)

#     def __call__(self, prediction, target_is_real):
#         if self.gan_mode in ['lsgan', 'vanilla']:
#             loss = self.loss(prediction, self.get_target_tensor(prediction, target_is_real))
#         elif self.gan_mode == 'wgangp':
#             loss = -prediction.mean() if target_is_real else prediction.mean()
#         return loss


# def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
#     if lambda_gp <= 0.0:
#         return 0.0, None
#     if type == 'real':
#         interpolatesv = real_data
#     elif type == 'fake':
#         interpolatesv = fake_data
#     elif type == 'mixed':
#         alpha = th.rand(real_data.shape[0], 1, device=device)
#         alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
#         interpolatesv = alpha * real_data + (1 - alpha) * fake_data
#     else:
#         raise NotImplementedError(f'{type} not implemented')
#     interpolatesv.requires_grad_(True)
#     disc_interpolates = netD(interpolatesv)
#     gradients = th.autograd.grad(
#         outputs=disc_interpolates, inputs=interpolatesv,
#         grad_outputs=th.ones(disc_interpolates.size(), device=device),
#         create_graph=True, retain_graph=True, only_inputs=True,
#     )[0]
#     gradients = gradients.view(real_data.size(0), -1)
#     gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
#     return gradient_penalty, gradients