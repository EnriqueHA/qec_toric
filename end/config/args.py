from dataclasses import dataclass, field
import torch

@dataclass
class Args():

    # Toric and noise
    error_rate: float = 0.17
    error_rates: list[float] | None = None  # if set, train on mix of error rates
    distance: int = 17
    seed: int | None = None
    norm: float | int = torch.inf

    # Management
    device: torch.device = field(
    default_factory = lambda: torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    ))

    save_model: bool = False
    checkpoint_dir: str = "checkpoints"
    
    # Model
    # Hyperparameters
    batch_size: int = 2048
    n_batches: int = 256
    n_epochs: int = 200
    epochs_warmup: int | None = None
    lr_muon: float = 3e-3
    lr_lion: float = 2e-4
    weight_decay: float = 1e-3
    init_channels: int = 128

    # Test
    model_path: str | None = None
    results_dir: str = "results"
    save_results: bool = True
    custom_result_name: str | None = None
    latency: bool = False