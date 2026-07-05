import math
import torch


def sample_t(
    n: int, T: float, eps_t: float, mode: str, device: torch.device
) -> torch.Tensor:
    """Sample diffusion timesteps t for training.

    mode='uniform'     : t ~ Uniform([eps_t, T])
    mode='log_uniform' : log(t) ~ Uniform([log(eps_t), log(T)]),
                         which over-samples small t where the score changes fastest.
    """
    u = torch.rand(n, device=device)
    if mode == "log_uniform":
        return torch.exp(u * (math.log(T) - math.log(eps_t)) + math.log(eps_t))
    return u * (T - eps_t) + eps_t

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().float() 
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def apply(self, model):
        """Load EMA weights into model for evaluation/generation."""
        model.load_state_dict({k: v.to(next(model.parameters()).device) 
                               for k, v in self.shadow.items()})