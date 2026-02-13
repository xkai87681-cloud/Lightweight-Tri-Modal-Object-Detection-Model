import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Logger:

    def __init__(
        self,
        log_dir: str = 'runs/logs',
        rank: int = 0,
        filename: str = 'train.log',
        use_tensorboard: bool = True,
        use_file_logging: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.rank = rank
        self.is_main_process = (rank == 0)

        if self.is_main_process:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = None
        if use_tensorboard and self.is_main_process:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            print(f"TensorBoard logging to {self.log_dir}")

        if use_file_logging and self.is_main_process:
            self._setup_file_logger(filename)
        else:
            self._setup_console_logger()

    def _setup_file_logger(self, filename: str):
        self.logger = logging.getLogger('MTL_Logger')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if self.logger.handlers:
            self.logger.handlers.clear()

        log_file = self.log_dir / filename
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info(f"File logging to {log_file}")

    def _setup_console_logger(self):
        self.logger = logging.getLogger(f'MTL_Logger_Rank{self.rank}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if self.logger.handlers:
            self.logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            f'[Rank {self.rank}] [%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def log_scalar(self, tag: str, value: Union[float, int, torch.Tensor], step: int):
        if not self.is_main_process:
            return

        if isinstance(value, torch.Tensor):
            value = value.item()

        if self.writer is not None:
            self.writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, tag_dict: Dict[str, Union[float, int, torch.Tensor]], step: int):
        if not self.is_main_process:
            return

        for tag, value in tag_dict.items():
            self.log_scalar(tag, value, step)

    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: int):
        if not self.is_main_process:
            return

        if self.writer is None:
            return

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if image.ndim == 3 and image.shape[-1] in [1, 3, 4]:
            image = image.permute(2, 0, 1)

        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        self.writer.add_image(tag, image, global_step=step)

    def log_images(self, tag: str, images: torch.Tensor, step: int):
        if not self.is_main_process:
            return

        if self.writer is None:
            return

        if images.dtype == torch.uint8:
            images = images.float() / 255.0

        self.writer.add_images(tag, images, global_step=step)

    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: int):
        if not self.is_main_process:
            return

        if self.writer is None:
            return

        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)

        self.writer.add_histogram(tag, values, global_step=step)

    def log_text(self, tag: str, text: str, step: int):
        if not self.is_main_process:
            return

        if self.writer is None:
            return

        self.writer.add_text(tag, text, global_step=step)

    def log_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]):
        if not self.is_main_process:
            return

        if self.writer is None:
            return

        hparam_dict_clean = {}
        for k, v in hparam_dict.items():
            if isinstance(v, (int, float, str, bool)):
                hparam_dict_clean[k] = v
            else:
                hparam_dict_clean[k] = str(v)

        self.writer.add_hparams(hparam_dict_clean, metric_dict)

    def log_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor):
        if not self.is_main_process:
            return

        if self.writer is None:
            return

        self.writer.add_graph(model, input_to_model)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            print("TensorBoard writer closed")

        if hasattr(self, 'logger'):
            for handler in self.logger.handlers:
                handler.close()

    def __del__(self):
        self.close()


_global_logger = None


def get_logger(
    log_dir: str = 'runs/logs',
    rank: int = 0,
    reset: bool = False
) -> Logger:
    global _global_logger

    if _global_logger is None or reset:
        _global_logger = Logger(log_dir=log_dir, rank=rank)

    return _global_logger


if __name__ == "__main__":
    print("Testing Logger...")

    logger = Logger(log_dir='runs/test_logs', rank=0)

    logger.log_scalar('train/loss', 0.5, step=1)
    logger.log_scalar('train/loss', 0.4, step=2)

    logger.log_scalars({
        'val/mAP': 0.75,
        'val/mIoU': 0.82,
        'val/acc': 0.90
    }, step=10)

    random_image = torch.rand(3, 224, 224)
    logger.log_image('test/image', random_image, step=1)

    logger.log_text('config', 'batch_size=8, lr=1e-4', step=0)

    logger.log_hparams(
        hparam_dict={'lr': 1e-4, 'batch_size': 8, 'epochs': 100},
        metric_dict={'best_mAP': 0.75, 'best_mIoU': 0.82}
    )

    logger.info("Training started")
    logger.warning("Learning rate is very small")
    logger.error("An error occurred!")

    logger.close()

    print("\n✅ Logger test passed!")
    print(f"Check TensorBoard: tensorboard --logdir=runs/test_logs")
