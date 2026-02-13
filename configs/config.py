import os
from pathlib import Path

class Config:

    PROJECT_ROOT = Path(__file__).parent.parent.absolute()

    DATASET_ROOT = PROJECT_ROOT

    BDD100K_ROOT = Path('/root/autodl-tmp/new_task/data/bdd100k')
    BDD100K_IMAGES_TRAIN = BDD100K_ROOT / 'images' / 'train'
    BDD100K_IMAGES_VAL = BDD100K_ROOT / 'images' / 'val'
    BDD100K_LABELS_DET_TRAIN = BDD100K_ROOT / 'labels' / 'detection' / 'train'
    BDD100K_LABELS_DET_VAL = BDD100K_ROOT / 'labels' / 'detection' / 'val'
    BDD100K_LABELS_SEG_TRAIN = BDD100K_ROOT / 'labels' / 'segmentation' / 'train'
    BDD100K_LABELS_SEG_VAL = BDD100K_ROOT / 'labels' / 'segmentation' / 'val'

    CITYSCAPES_ROOT = BDD100K_ROOT  # 向后兼容
    CITYSCAPES_IMAGES_TRAIN = BDD100K_IMAGES_TRAIN
    CITYSCAPES_IMAGES_VAL = BDD100K_IMAGES_VAL
    CITYSCAPES_LABELS_TRAIN = BDD100K_LABELS_DET_TRAIN  # 检测标注
    CITYSCAPES_LABELS_VAL = BDD100K_LABELS_DET_VAL

    PA100K_ROOT = DATASET_ROOT / 'PA100K'
    PA100K_IMAGES = PA100K_ROOT / 'release_data' / 'release_data'
    PA100K_ANNOTATION = PA100K_ROOT / 'annotation.mat'
    PA100K_TRAIN_CSV = PA100K_ROOT / 'annotations' / 'train.csv'
    PA100K_VAL_CSV = PA100K_ROOT / 'annotations' / 'val.csv'
    USE_PA100K_CSV = False  # 设为False，使用MAT格式

    OUTPUT_DIR = PROJECT_ROOT / 'runs'
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    LOG_DIR = OUTPUT_DIR / 'logs'
    VIS_DIR = OUTPUT_DIR / 'visualizations'

    BACKBONE = 'mobilenetv3_large'
    BACKBONE_PRETRAINED = True  # 🚀 使用预训练权重
    BACKBONE_PRETRAINED_TYPE = 'imagenet'  # 🔧 暂时使用 ImageNet 预训练（更稳定）
    BACKBONE_OUT_CHANNELS = [40, 112, 960]  # C3, C4, C5

    NECK_CHANNELS = 256
    NECK_OUT_CHANNELS = [128, 256, 512]  # P3, P4, P5

    DET_NUM_CLASSES = 2  # person, car
    DET_ANCHORS_PER_LEVEL = 3
    DET_IOU_THRESHOLD = 0.5
    DET_CONF_THRESHOLD = 0.15  # 降低阈值提高召回（0.25→0.15）
    DET_NMS_THRESHOLD = 0.35  # 降低NMS阈值（0.40→0.35）保留更多Person（0.45→0.40）
    DET_MAX_DETECTIONS = 500  # 增加最大检测数（300→500）

    ATTR_NUM_CLASSES = 26  # PA100K 26个行人属性
    ATTR_ROI_SIZE = 7  # RoIAlign 输出大小 7x7
    ATTR_ROI_SPATIAL_SCALE = [1/8, 1/16, 1/32]  # P3, P4, P5 相对原图的缩放
    ATTR_HIDDEN_DIM = 512
    ATTR_DROPOUT = 0.5  # 🛡️ 增加Dropout防止过拟合（0.3→0.5）

    SEG_NUM_CLASSES = 3  # background=0, road=1, lane=2
    SEG_USE_P3 = True
    SEG_USE_P4 = True
    SEG_USE_P5 = False  # 强制禁用
    SEG_IGNORE_INDEX = 255  # 仅用于标注中真正需要忽略的像素（如边界）

    USE_TASK_ATTENTION = True  # 是否启用任务特定注意力机制
    ATTENTION_REDUCTION = 16  # 注意力模块的通道压缩比例
    USE_ATTENTION_FUSION = True  # 是否使用特征融合（原始特征+注意力特征）

    IMAGE_SIZE = 640  # 输入图像大小
    TRAIN_AUGMENT = True

    AUG_BRIGHTNESS_LIMIT = 0.2
    AUG_CONTRAST_LIMIT = 0.2
    AUG_HUE_SHIFT_LIMIT = 20
    AUG_SAT_SHIFT_LIMIT = 30
    AUG_VAL_SHIFT_LIMIT = 20
    AUG_BLUR_LIMIT = 7
    AUG_NOISE_VAR_LIMIT = (10.0, 50.0)

    USE_MOSAIC = True
    MOSAIC_PROB = 1.0  # 🔧 提高到100%防止过拟合
    USE_MIXUP = True  # 🔧 启用MixUp增加正则化
    MIXUP_ALPHA = 0.15  # MixUp混合系数（轻度混合）

    EPOCHS = 100  # 增加训练时间（50→100）让检测充分收敛，减少epoch数（100→50）
    BATCH_SIZE = 24  # 调整batch size（16→20）平衡稳定性和泛化
    VAL_BATCH_SIZE = 48  # 验证集batch size
    NUM_WORKERS = 8  # 提高数据加载速度（4→8）
    PIN_MEMORY = True  # 启用pin memory加速

    OPTIMIZER = 'adamw'  # 'sgd', 'adam', or 'adamw'
    LR = 5e-5  # 🔧 降低学习率提高稳定性（1e-4→5e-5）
    WEIGHT_DECAY = 0.05  # 🔧 增加正则化防止过拟合（0.01→0.05）
    MOMENTUM = 0.937  # 仅用于 SGD

    SCHEDULER = 'cosine'  # 使用标准 cosine
    WARMUP_EPOCHS = 3      # Warmup epochs
    WARMUP_LR = 1e-5       # Warmup 起始学习率
    LR_MIN = 1e-6         # cosine scheduler的最小学习率
    LR_STEP_SIZE = 15  # step scheduler的步长（30→15）
    LR_GAMMA = 0.1  # step/multistep scheduler的衰减率
    LR_MILESTONES = [30, 60, 90]  # multistep scheduler的里程碑

    T_0 = 10  # 初始周期长度（epochs）
    T_MULT = 2  # 周期倍增因子
    ETA_MIN = 1e-6  # 最小学习率

    USE_AMP = True

    GRAD_CLIP_NORM = 10.0  # 🔧 降低梯度裁剪阈值（20.0→10.0）更稳定
    GRAD_CLIP_SEG = 5.0    # Seg Head 梯度裁剪（降低：20.0→5.0）
    GRAD_CLIP_ATTR = 5.0   # Attr Head 单独的梯度裁剪

    USE_EMA = True
    EMA_DECAY = 0.9999

    USE_INTERLEAVED_SAMPLER = True

    TASK_BATCH_RATIOS = {
        'detection': 1.5,      # 70000 × 1.5 = 105000 samples/epoch (提高采样) (提高采样)
        'attribute': 0.9,      # 80000 × 0.9 = 72000 samples/epoch ✅
        'segmentation': 1.5    # 7000 × 1.5 = 10500 samples/epoch (降低：2.0→1.5)
    }

    ATTR_START_EPOCH = 20  # 🔧 Attribute 延迟到 epoch 20 开始
    CURRICULUM_EPOCH = 20  # trainer.py使用的参数名

    USE_STAGE_TRAINING = True
    STAGE_1_EPOCHS = 10   # Stage 1: Detection Only (充分学习检测)
    STAGE_2_EPOCHS = 20   # Stage 2: Detection + Segmentation

    STAGE_1_WEIGHTS = {
        'detection': 1.0,      # 只训练检测
        'attribute': 0.0,
        'segmentation': 0.0
    }

    STAGE_2_WEIGHTS = {
        'detection': 1.0,      # 检测权重最高
        'attribute': 0.0,
        'segmentation': 0.2    # 分割权重较低，避免干扰检测
    }

    STAGE_3_WEIGHTS = {
        'detection': 1.0,      # 检测仍然是主任务
        'attribute': 0.3,      # 属性权重较低
        'segmentation': 0.3    # 分割权重较低
    }


    USE_PROGRESSIVE_UNFREEZING = False # 禁用（数据量够了）
    FREEZE_BACKBONE_EPOCHS = 0         # 不冻结
    FREEZE_NECK_EPOCHS = 0             # 不冻结
    UNFREEZE_SCHEDULE = 'all_at_once'  # 'all_at_once' 或 'gradual'

    USE_PCGRAD = False  # PCGrad（默认关闭）
    USE_GRADNORM = False  # GradNorm（默认关闭）
    GRADNORM_ALPHA = 1.5  # GradNorm 超参数

    LOSS_WEIGHT_DET = 1.0      # Detection（主任务，权重最高）
    LOSS_WEIGHT_ATTR = 0.2     # Attribute（辅助任务，权重低）
    LOSS_WEIGHT_SEG = 0.2      # Segmentation（辅助任务，权重低）

    USE_AUTO_LOSS = True  # 启用自动损失加权（基于不确定性）
    AUTO_LOSS_INIT_LOG_VARS = [0.0, 0.0, 0.0]  # 初始log_var值 [det, attr, seg]
    AUTO_LOSS_CLAMP_RANGE = (-10, 10)  # log_var的裁剪范围，防止数值不稳定

    DET_BOX_LOSS_WEIGHT = 3.0      # BBox loss权重（提高：2.0→3.0）（降低）
    DET_CLS_LOSS_WEIGHT = 1.0      # Classification loss权重（提高）
    DET_OBJ_LOSS_WEIGHT = 1.0      # Objectness loss权重

    DET_FOCAL_LOSS_GAMMA = 2.0  # 提高gamma（1.5→2.0）更关注难样本
    DET_FOCAL_LOSS_ALPHA = 0.25

    ATTR_USE_FOCAL_LOSS = False  # 使用 Asymmetric Focal Loss
    ATTR_FOCAL_GAMMA_NEG = 4
    ATTR_FOCAL_GAMMA_POS = 1
    ATTR_CLIP = 0.05
    ATTR_LABEL_SMOOTHING = 0.1  # 🛡️ Label Smoothing防止过拟合

    SEG_CE_WEIGHT = 1.0
    SEG_DICE_WEIGHT = 1.0

    SEG_USE_FOCAL_LOSS = True  # 启用Focal Loss替代CE Loss
    SEG_FOCAL_ALPHA = [1.0, 3.0, 10.0]  # Per-class权重 [background, road, lane]
    SEG_FOCAL_GAMMA = 2.0  # Focusing参数，越大越关注难样本

    VAL_INTERVAL = 1  # 每隔几个 epoch 验证一次
    SAVE_INTERVAL = 5  # 每隔几个 epoch 保存一次
    SAVE_BEST_ONLY = True

    USE_EARLY_STOPPING = False  # 🔧 暂时禁用早停，让训练完整运行
    EARLY_STOPPING_PATIENCE = 15  # 早停耐心值
    EARLY_STOPPING_MIN_DELTA = 0.0001  # 改善的最小阈值
    EARLY_STOPPING_MONITOR = 'composite_loss'  # 监控的指标：'composite_loss' 或 'val_loss'
    EARLY_STOPPING_MODE = 'min'  # 'min' 或 'max'，loss越小越好所以用min

    OVERFIT_THRESHOLD = 2.0  # Train/Val loss比率阈值（超过2.0x发出警告）
    OVERFIT_ACTION = 'warn'  # 'warn' 或 'stop' 或 'reduce_lr'

    METRIC_WEIGHTS = {
        'det_map': 0.4,
        'attr_map': 0.3,
        'seg_miou': 0.3
    }

    DISTRIBUTED = False  # 是否使用分布式训练
    DIST_BACKEND = 'nccl'  # 'nccl' for GPU, 'gloo' for CPU
    DIST_URL = 'env://'  # 使用环境变量初始化
    WORLD_SIZE = 1
    RANK = 0
    LOCAL_RANK = 0

    SEED = 42
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    LOG_INTERVAL = 10  # 每隔几个 iter 打印一次
    VIS_INTERVAL = 50  # 每隔几个 iter 可视化一次

    RESUME = None  # checkpoint 路径，None 表示从头训练

    PRETRAINED_MODEL = None  # 预训练模型路径
    LOAD_BACKBONE_ONLY = False  # 仅加载 backbone 权重

    @classmethod
    def display(cls):
        print("=" * 80)
        print("YOLOv8-MobileNetV3 Multi-Task Learning Configuration")
        print("=" * 80)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key:30s}: {value}")
        print("=" * 80)

    @classmethod
    def update(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                print(f"Warning: {key} is not a valid config key")

    @classmethod
    def create_dirs(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.VIS_DIR.mkdir(parents=True, exist_ok=True)

cfg = Config()
