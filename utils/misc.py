import os
import random
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def get_device(device_str: str = 'cuda') -> torch.device:
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def create_dirs(*dirs: str):
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group['lr']


def clip_gradients(
    model: nn.Module,
    max_norm: float = 10.0,
    norm_type: float = 2.0
) -> float:
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        norm_type=norm_type
    )
    return total_norm.item()


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str,
    filename: str = 'checkpoint.pt'
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / filename
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    if is_best:
        best_path = checkpoint_dir / 'best.pt'
        torch.save(state, best_path)
        print(f"Best model saved to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = None
) -> Dict[str, Any]:
    print(f"Loading checkpoint from {checkpoint_path}")

    if device is None:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'best_metric' in checkpoint:
        print(f"  Best metric: {checkpoint['best_metric']:.4f}")

    return checkpoint


def print_config(config: Dict[str, Any], title: str = "Configuration"):
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

    for key, value in config.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key:30s}: {sub_value}")
        else:
            print(f"{key:30s}: {value}")

    print("=" * 80 + "\n")


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def model_info(model: nn.Module, verbose: bool = False):
    n_p = count_parameters(model, trainable_only=False)  # 总参数
    n_t = count_parameters(model, trainable_only=True)   # 可训练参数
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad and len(x.shape) > 1)  # gradients

    print(f"\nModel Summary:")
    print(f"  Total parameters: {n_p:,}")
    print(f"  Trainable parameters: {n_t:,}")
    print(f"  Gradient parameters: {n_g:,}")
    print(f"  Layers: {len(list(model.modules()))}")

    if verbose:
        print("\nLayer Details:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules
                params = sum(p.numel() for p in module.parameters())
                print(f"  {name:50s} {params:>12,}")


def check_file(file_path: str) -> bool:
    return Path(file_path).exists()


def make_divisible(x: int, divisor: int = 8) -> int:
    return int(np.ceil(x / divisor) * divisor)


if __name__ == "__main__":
    print("Testing misc utilities...")

    set_seed(42)

    device = get_device('cuda')

    create_dirs('runs/test1', 'runs/test2')

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    n_params = count_parameters(model)
    print(f"\nModel parameters: {n_params:,}")

    model_info(model, verbose=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr = get_lr(optimizer)
    print(f"\nCurrent learning rate: {lr}")

    config = {
        'model': 'YOLOv8',
        'backbone': 'MobileNetV3',
        'training': {
            'batch_size': 8,
            'lr': 1e-4,
            'epochs': 100
        }
    }
    print_config(config, title="Test Configuration")

    t0 = time_sync()
    time.sleep(0.1)
    t1 = time_sync()
    print(f"\nElapsed time: {t1 - t0:.4f}s")

    print(f"\nmake_divisible(23, 8) = {make_divisible(23, 8)}")

    print("\n✅ Misc utilities test passed!")
