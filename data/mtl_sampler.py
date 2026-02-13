
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Iterator
from torch.utils.data import Sampler

try:
    import torch
except ImportError:
    torch = None

from configs.config import cfg


class InterleavedSampler(Sampler):

    def __init__(
        self,
        dataset_sizes: Dict[str, int],
        task_ratios: Dict[str, float] = None,
        shuffle: bool = True,
        seed: int = None,
        drop_last: bool = False
    ):
        self.dataset_sizes = dataset_sizes
        self.shuffle = shuffle
        self.seed = seed if seed is not None else cfg.SEED
        self.drop_last = drop_last
        self.epoch = 0

        self.tasks = list(dataset_sizes.keys())
        self.num_tasks = len(self.tasks)

        if task_ratios is None:
            if hasattr(cfg, 'TASK_BATCH_RATIOS'):
                task_ratios = cfg.TASK_BATCH_RATIOS
            else:
                task_ratios = {task: 1.0 for task in self.tasks}

        self.task_ratios = task_ratios

        self.effective_sizes = {}
        for task in self.tasks:
            base_size = dataset_sizes[task]
            ratio = task_ratios.get(task, 1.0)
            self.effective_sizes[task] = int(base_size * ratio)

        self.total_samples = sum(self.effective_sizes.values())

        print(f"InterleavedSampler initialized:")
        print(f"  Dataset sizes: {dataset_sizes}")
        print(f"  Task ratios: {task_ratios}")
        print(f"  Effective sizes: {self.effective_sizes}")
        print(f"  Total samples: {self.total_samples}")

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        if self.seed is not None:
            random.seed(self.seed + self.epoch)
            np.random.seed(self.seed + self.epoch)
            if torch is not None:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)

        task_indices = {}
        for task in self.tasks:
            base_size = self.dataset_sizes[task]
            effective_size = self.effective_sizes[task]

            indices = list(range(base_size))
            if self.shuffle:
                random.shuffle(indices)

            if effective_size > base_size:
                repeats = effective_size // base_size
                remainder = effective_size % base_size
                indices = indices * repeats + indices[:remainder]
            elif effective_size < base_size:
                indices = indices[:effective_size]

            task_indices[task] = indices

        interleaved_samples = []

        max_len = max(len(indices) for indices in task_indices.values())

        for i in range(max_len):
            task_order = self.tasks.copy()
            if self.shuffle:
                random.shuffle(task_order)

            for task in task_order:
                task_idx_list = task_indices[task]
                if i < len(task_idx_list):
                    sample_idx = task_idx_list[i]
                    interleaved_samples.append((task, sample_idx))

        if self.shuffle:
            random.shuffle(interleaved_samples)

        return iter(interleaved_samples)

    def __len__(self) -> int:
        return self.total_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class MultiTaskBatchSampler(Sampler):

    def __init__(
        self,
        dataset_sizes: Dict[str, int],
        batch_size: int,
        task_ratios: Dict[str, float] = None,
        shuffle: bool = True,
        seed: int = None
    ):
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed if seed is not None else cfg.SEED

        self.tasks = ['detection', 'attribute', 'segmentation']

        if task_ratios is None:
            if hasattr(cfg, 'TASK_BATCH_RATIOS'):
                task_ratios = cfg.TASK_BATCH_RATIOS
            else:
                task_ratios = {task: 1.0 for task in self.tasks}

        self.task_ratios = task_ratios

        self.task_offsets = {}
        offset = 0
        for task in self.tasks:
            self.task_offsets[task] = offset
            offset += dataset_sizes[task]

        print(f"MultiTaskBatchSampler initialized:")
        print(f"  Task offsets: {self.task_offsets}")

    def __iter__(self) -> Iterator[List[int]]:
        if self.seed is not None:
            random.seed(self.seed)

        all_batches = []

        for task in self.tasks:
            size = self.dataset_sizes[task]
            ratio = self.task_ratios.get(task, 1.0)
            effective_size = int(size * ratio)
            offset = self.task_offsets[task]

            local_indices = list(range(size))
            if self.shuffle:
                random.shuffle(local_indices)

            if effective_size > size:
                repeats = effective_size // size
                remainder = effective_size % size
                local_indices = local_indices * repeats + local_indices[:remainder]
            elif effective_size < size:
                local_indices = local_indices[:effective_size]

            global_indices = [offset + local_idx for local_idx in local_indices]

            num_batches = (len(global_indices) + self.batch_size - 1) // self.batch_size
            for i in range(num_batches):
                batch_indices = global_indices[i * self.batch_size : (i + 1) * self.batch_size]
                all_batches.append(batch_indices)

        if self.shuffle:
            random.shuffle(all_batches)

        return iter(all_batches)

    def __len__(self) -> int:
        total = 0
        for task in self.tasks:
            size = self.dataset_sizes[task]
            ratio = self.task_ratios.get(task, 1.0)
            effective_size = int(size * ratio)
            num_batches = (effective_size + self.batch_size - 1) // self.batch_size
            total += num_batches
        return total


if __name__ == '__main__':
    print("="*80)
    print("Testing InterleavedSampler")
    print("="*80)

    dataset_sizes = {
        'detection': 100,
        'attribute': 60,
        'segmentation': 120
    }

    task_ratios = {
        'detection': 1.0,
        'attribute': 0.5,  # 属性任务只采样一半
        'segmentation': 1.0
    }

    sampler = InterleavedSampler(
        dataset_sizes=dataset_sizes,
        task_ratios=task_ratios,
        shuffle=True,
        seed=42
    )

    print(f"\nTotal samples: {len(sampler)}")

    samples = list(sampler)[:30]
    print(f"\nFirst 30 samples:")
    task_counts = {'detection': 0, 'attribute': 0, 'segmentation': 0}
    for i, (task, idx) in enumerate(samples):
        print(f"  {i:2d}: task={task:12s}, idx={idx:3d}")
        task_counts[task] += 1

    print(f"\nTask distribution in first 30 samples:")
    for task, count in task_counts.items():
        print(f"  {task}: {count}")

    print("\n" + "="*80)
    print("Testing MultiTaskBatchSampler")
    print("="*80)

    batch_sampler = MultiTaskBatchSampler(
        dataset_sizes=dataset_sizes,
        batch_size=8,
        task_ratios=task_ratios,
        shuffle=True,
        seed=42
    )

    print(f"\nTotal batches: {len(batch_sampler)}")

    batches = list(batch_sampler)[:10]
    print(f"\nFirst 10 batches:")
    for i, (task, indices) in enumerate(batches):
        print(f"  Batch {i}: task={task:12s}, size={len(indices)}, indices={indices[:5]}...")

    print("\n✅ Sampler tests passed!")
