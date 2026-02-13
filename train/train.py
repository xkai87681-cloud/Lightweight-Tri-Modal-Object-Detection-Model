import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import cfg
from models.mtl_model import MTLModel
from losses.det_loss_ultralytics import UltralyticsDetectionLoss
from losses.attr_loss import AttributeLoss
from losses.seg_loss import SegmentationLoss
from data.bdd100k_det import BDD100KDetection, collate_fn_det
from data.pa100k_dataset import PA100KDataset, collate_fn_attr
from data.bdd100k_seg import BDD100KSegmentation, collate_fn_seg
from data.mtl_sampler import InterleavedSampler
from data.transforms import get_train_transform, get_val_transform
from train.trainer import MTLTrainer
from train.distributed import init_distributed, is_main_process


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Task Learning Training')

    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备: cuda or cpu')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch_size', type=int, default=None, help='batch大小（覆盖配置文件）')
    parser.add_argument('--lr', type=float, default=None, help='学习率（覆盖配置文件）')

    parser.add_argument('--use_amp', action='store_true', default=True, help='使用AMP混合精度训练')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false', help='禁用AMP')
    parser.add_argument('--use_ema', action='store_true', default=True, help='使用EMA')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false', help='禁用EMA')
    parser.add_argument('--use_pcgrad', action='store_true', default=False, help='使用PCGrad')
    parser.add_argument('--use_gradnorm', action='store_true', default=False, help='使用GradNorm')

    parser.add_argument('--distributed', action='store_true', default=False, help='使用DDP分布式训练')

    parser.add_argument('--curriculum_epoch', type=int, default=None, help='Curriculum Learning起始epoch')

    parser.add_argument('--log_dir', type=str, default='runs', help='TensorBoard日志目录')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='检查点保存目录')
    parser.add_argument('--save_interval', type=int, default=5, help='保存间隔（epoch）')
    parser.add_argument('--val_interval', type=int, default=1, help='验证间隔（epoch）')

    return parser.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_dict):
        self.datasets = datasets_dict

        self.tasks = ['detection', 'attribute', 'segmentation']

        self.task_sizes = {task: len(datasets_dict[task]) for task in self.tasks}
        self.total_size = sum(self.task_sizes.values())

        self.index_map = []
        for task in self.tasks:
            for i in range(self.task_sizes[task]):
                self.index_map.append((task, i))

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            task, local_idx = idx
        else:
            task, local_idx = self.index_map[idx]

        sample = self.datasets[task][local_idx]
        if isinstance(sample, dict):
            sample['task'] = task
        return task, sample


def mtl_collate_fn(batch):
    if len(batch) == 0:
        return None

    tasks = [item[0] for item in batch]
    task = tasks[0]

    if not all(t == task for t in tasks):
        raise ValueError(f"Batch contains mixed tasks: {set(tasks)}")

    samples = [item[1] for item in batch]

    if task == 'detection':
        return ('detection', collate_fn_det(samples))
    elif task == 'attribute':
        return ('attribute', collate_fn_attr(samples))
    elif task == 'segmentation':
        return ('segmentation', collate_fn_seg(samples))
    else:
        raise ValueError(f"Unknown task: {task}")


def build_dataloaders(split: str = 'train', use_sampler: bool = True):
    if split == 'train':
        transform_det = get_train_transform(image_size=cfg.IMAGE_SIZE, task='detection')
        transform_attr = get_train_transform(image_size=cfg.IMAGE_SIZE, task='attribute')
        transform_seg = get_train_transform(image_size=cfg.IMAGE_SIZE, task='segmentation')
    else:
        transform_det = get_val_transform(image_size=cfg.IMAGE_SIZE, task='detection')
        transform_attr = get_val_transform(image_size=cfg.IMAGE_SIZE, task='attribute')
        transform_seg = get_val_transform(image_size=cfg.IMAGE_SIZE, task='segmentation')

    datasets = {}

    datasets['detection'] = BDD100KDetection(
        split=split,
        transform=transform_det,
        image_size=cfg.IMAGE_SIZE
    )

    datasets['attribute'] = PA100KDataset(
        split=split,
        transform=transform_attr,
        image_size=cfg.IMAGE_SIZE
    )

    datasets['segmentation'] = BDD100KSegmentation(
        split=split,
        transform=transform_seg,
        image_size=cfg.IMAGE_SIZE
    )

    print(f"\n{split.upper()} Datasets:")
    for task, ds in datasets.items():
        print(f"  {task}: {len(ds)} samples")

    batch_size = cfg.BATCH_SIZE if split == 'train' else cfg.VAL_BATCH_SIZE

    if use_sampler and split == 'train' and cfg.USE_INTERLEAVED_SAMPLER:
        print(f"\n使用MultiTaskBatchSampler进行多任务交错采样")

        dataset_sizes = {task: len(ds) for task, ds in datasets.items()}

        mtl_dataset = MultiTaskDataset(datasets)

        from data.mtl_sampler import MultiTaskBatchSampler

        batch_sampler = MultiTaskBatchSampler(
            dataset_sizes=dataset_sizes,
            batch_size=batch_size,
            task_ratios=cfg.TASK_BATCH_RATIOS,
            shuffle=True,
            seed=cfg.SEED
        )

        loader = DataLoader(
            mtl_dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=mtl_collate_fn
        )

        print(f"\n统一DataLoader:")
        print(f"  Total batches: {len(loader)}")
        print(f"  Batch size: {batch_size}")

        return loader

    else:
        loaders = {}

        loaders['detection'] = DataLoader(
            datasets['detection'],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=collate_fn_det
        )

        loaders['attribute'] = DataLoader(
            datasets['attribute'],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=collate_fn_attr
        )

        loaders['segmentation'] = DataLoader(
            datasets['segmentation'],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=collate_fn_seg
        )

        print(f"\n{split.upper()} DataLoaders:")
        for task, loader in loaders.items():
            print(f"  {task}: {len(loader)} batches, {len(loader.dataset)} samples")

        return loaders


def build_model(device):
    model = MTLModel(
        num_det_classes=cfg.DET_NUM_CLASSES,
        num_attr_classes=cfg.ATTR_NUM_CLASSES,
        num_seg_classes=cfg.SEG_NUM_CLASSES,
        backbone_pretrained=cfg.BACKBONE_PRETRAINED,
        backbone_pretrained_type=getattr(cfg, 'BACKBONE_PRETRAINED_TYPE', 'detection'),
        use_task_attention=getattr(cfg, 'USE_TASK_ATTENTION', False),
        attention_reduction=getattr(cfg, 'ATTENTION_REDUCTION', 16),
        use_attention_fusion=getattr(cfg, 'USE_ATTENTION_FUSION', True),
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel:")
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    if getattr(cfg, 'USE_TASK_ATTENTION', False):
        print(f"  Task-Specific Attention: ENABLED (reduction={getattr(cfg, 'ATTENTION_REDUCTION', 16)})")

    return model


def build_optimizer(model):
    if cfg.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.LR,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY
        )
    elif cfg.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY
        )
    elif cfg.OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.OPTIMIZER}")

    print(f"\nOptimizer: {cfg.OPTIMIZER}")
    print(f"  LR: {cfg.LR}")
    print(f"  Weight decay: {cfg.WEIGHT_DECAY}")

    return optimizer


def build_scheduler(optimizer):
    if cfg.SCHEDULER == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.EPOCHS - cfg.WARMUP_EPOCHS,  # 减去warmup的epoch数
            eta_min=cfg.LR_MIN
        )

        if cfg.WARMUP_EPOCHS > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=cfg.WARMUP_LR / cfg.LR,  # 起始因子
                end_factor=1.0,  # 结束因子
                total_iters=cfg.WARMUP_EPOCHS
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[cfg.WARMUP_EPOCHS]
            )
        else:
            scheduler = main_scheduler
    elif cfg.SCHEDULER == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.LR_STEP_SIZE,
            gamma=cfg.LR_GAMMA
        )
    elif cfg.SCHEDULER == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.LR_MILESTONES,
            gamma=cfg.LR_GAMMA
        )
    else:
        scheduler = None

    if scheduler is not None:
        print(f"\nScheduler: {cfg.SCHEDULER}")
    else:
        print("\nNo scheduler")

    return scheduler


def build_loss_functions(attr_pos_weight=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_fns = {
        'detection': UltralyticsDetectionLoss(
            num_classes=cfg.DET_NUM_CLASSES,
            reg_max=16,
            device=device
        ),
        'attribute': AttributeLoss(
            num_attrs=cfg.ATTR_NUM_CLASSES,
            pos_weight=attr_pos_weight,  # 🔧 添加 pos_weight 处理类别不平衡
            label_smoothing=cfg.ATTR_LABEL_SMOOTHING
        ),
        'segmentation': SegmentationLoss(
            num_classes=cfg.SEG_NUM_CLASSES,
            ignore_index=255,
            use_focal_loss=cfg.SEG_USE_FOCAL_LOSS,
            focal_alpha=cfg.SEG_FOCAL_ALPHA,
            focal_gamma=cfg.SEG_FOCAL_GAMMA
        )
    }

    print("\nLoss Functions:")
    for task, loss_fn in loss_fns.items():
        print(f"  {task}: {loss_fn.__class__.__name__}")
    print("  ✓ Using Ultralytics YOLOv8 Detection Loss (TAL + CIoU + Multi-scale)")
    if attr_pos_weight is not None:
        print(f"  ✓ Attribute Loss with pos_weight for class imbalance")

    return loss_fns


def main():
    args = parse_args()

    set_seed(cfg.SEED)

    if args.distributed:
        init_distributed()

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("\nUsing CPU")

    print("\n" + "=" * 80)
    print("Building DataLoaders...")
    print("=" * 80)
    train_loader = build_dataloaders(split='train', use_sampler=True)  # 训练时使用采样器
    val_loaders = build_dataloaders(split='val', use_sampler=False)    # 验证时用独立loader

    print("\n" + "=" * 80)
    print("Computing Attribute pos_weight...")
    print("=" * 80)
    attr_dataset = PA100KDataset(split='train', transform=None, image_size=cfg.IMAGE_SIZE)
    attr_pos_weight = attr_dataset.compute_pos_weight()
    print(f"  Attribute pos_weight computed for {cfg.ATTR_NUM_CLASSES} classes")
    print(f"  pos_weight range: [{attr_pos_weight.min():.2f}, {attr_pos_weight.max():.2f}]")

    print("\n" + "=" * 80)
    print("Building Model...")
    print("=" * 80)
    model = build_model(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank] if hasattr(args, 'local_rank') else None
        )
        print("Model wrapped with DDP")

    print("\n" + "=" * 80)
    print("Building Optimizer and Scheduler...")
    print("=" * 80)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    print("\n" + "=" * 80)
    print("Building Loss Functions...")
    print("=" * 80)
    loss_fns = build_loss_functions(attr_pos_weight=attr_pos_weight)

    for task_name, loss_fn in loss_fns.items():
        loss_fns[task_name] = loss_fn.to(device)
    print(f"  ✓ Loss functions moved to {device}")

    print("\n" + "=" * 80)
    print("Creating Trainer...")
    print("=" * 80)
    trainer = MTLTrainer(
        model=model,
        train_loaders=train_loader,  # 注意：现在是单个loader
        val_loaders=val_loaders,     # 验证仍是dict
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fns=loss_fns,
        device=device,
        use_amp=args.use_amp,
        use_ema=args.use_ema,
        use_pcgrad=args.use_pcgrad,
        use_gradnorm=args.use_gradnorm,
        log_dir=args.log_dir,
        save_dir=args.save_dir
    )

    start_epoch = 0
    if args.resume:
        trainer.load_checkpoint(args.resume)
        start_epoch = trainer.current_epoch + 1

    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)

    epochs = args.epochs if args.epochs is not None else cfg.EPOCHS

    for epoch in range(start_epoch, epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'=' * 80}")

        if epoch + 1 == cfg.CURRICULUM_EPOCH and epoch > 0:
            print(f"\n🔄 Stage 3 started (Attribute task begins)")
            print(f"   Resetting best_metric to ensure best.pt includes Attribute training")
            trainer.best_metric = float('inf')
            if trainer.early_stopping is not None:
                trainer.early_stopping.reset()

        train_losses = trainer.train_epoch(epoch)

        print(f"\nTrain Losses:")
        for task, loss in train_losses.items():
            print(f"  {task}: {loss:.4f}")

        if (epoch + 1) % args.val_interval == 0:
            print(f"\nValidating...")
            val_losses = trainer.validate(epoch)

            print(f"\nValidation Losses:")
            for task, loss in val_losses.items():
                print(f"  {task}: {loss:.4f}")

            composite_loss = (
                val_losses.get('detection', 0.0) * cfg.LOSS_WEIGHT_DET +
                val_losses.get('attribute', 0.0) * cfg.LOSS_WEIGHT_ATTR +
                val_losses.get('segmentation', 0.0) * cfg.LOSS_WEIGHT_SEG
            )
            print(f"\nComposite Loss: {composite_loss:.4f}")

            is_best = composite_loss < trainer.best_metric if trainer.best_metric > 0 else True

            if is_best:
                trainer.best_metric = composite_loss
                print(f"✅ New best composite loss: {composite_loss:.4f}")

            if trainer.early_stopping is not None:
                should_stop = trainer.early_stopping(composite_loss, epoch + 1)

                if should_stop:
                    print(f"\n{'=' * 80}")
                    print(f"⛔ Early stopping triggered at epoch {epoch + 1}")
                    print(f"   Best composite loss: {trainer.early_stopping.best_score:.4f} at epoch {trainer.early_stopping.best_epoch}")
                    print(f"{'=' * 80}")

                    trainer.save_checkpoint(epoch, is_best=False)
                    break

        else:
            is_best = False

        if (epoch + 1) % args.save_interval == 0:
            trainer.save_checkpoint(epoch, is_best=is_best, periodic=True)
        elif is_best:
            trainer.save_checkpoint(epoch, is_best=True)

    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)

    trainer.close()


if __name__ == "__main__":
    main()
