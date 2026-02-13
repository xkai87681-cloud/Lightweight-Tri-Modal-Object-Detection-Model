import os
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional


def init_distributed(backend='nccl', init_method='env://'):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        print("Not using distributed mode")
        return False

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )

    dist.barrier()

    print(f"Distributed training initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    return True


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier():
    if is_distributed():
        dist.barrier()


def reduce_dict(input_dict: Dict[str, torch.Tensor], average: bool = True) -> Dict[str, torch.Tensor]:
    if not is_distributed():
        return input_dict

    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []

        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)

        dist.all_reduce(values)

        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    if not is_distributed():
        return tensor

    world_size = get_world_size()
    if world_size < 2:
        return tensor

    with torch.no_grad():
        rt = tensor.clone()
        dist.all_reduce(rt)
        if average:
            rt /= world_size

    return rt


def gather_tensors(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if not is_distributed():
        return tensor

    world_size = get_world_size()
    if world_size < 2:
        return tensor

    if is_main_process():
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    else:
        gathered = None

    dist.gather(tensor, gather_list=gathered, dst=0)

    if is_main_process():
        gathered = torch.cat(gathered, dim=0)

    return gathered


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def cleanup():
    if is_distributed():
        dist.destroy_process_group()


class DistributedSampler:
    def __init__(self, sampler, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0

        self.total_size = len(sampler)
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(self.epoch)

        indices = list(self.sampler)

        indices = indices[:self.num_samples * self.num_replicas]

        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


if __name__ == "__main__":
    print("Testing distributed utilities (non-distributed mode):")

    print(f"Rank: {get_rank()}")
    print(f"World size: {get_world_size()}")
    print(f"Is main process: {is_main_process()}")
    print(f"Is distributed: {is_distributed()}")

    test_dict = {
        'loss': torch.tensor(1.5),
        'acc': torch.tensor(0.85)
    }

    reduced = reduce_dict(test_dict)
    print(f"\nOriginal dict: {test_dict}")
    print(f"Reduced dict: {reduced}")

    tensor = torch.tensor([1.0, 2.0, 3.0])
    reduced_tensor = reduce_tensor(tensor)
    print(f"\nOriginal tensor: {tensor}")
    print(f"Reduced tensor: {reduced_tensor}")

    print("\n✅ Distributed utilities test passed (non-distributed mode)!")
