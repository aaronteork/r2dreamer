from dataclasses import dataclass
import torch


@dataclass(frozen=True, kw_only=True)
class BaseConfig:
    seed: int = 0
    device: torch.device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
    is_training: bool = True
