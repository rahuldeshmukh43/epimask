from yacs.config import CfgNode as CN
_CN = CN()


_CN.LOFTR = CN() # legacy for importing some LoFTR modules

##############  ↓  EpiMask Pipeline  ↓  ##############
_CN.EPIMASK = CN()
_CN.EPIMASK.BACKBONE_TYPE = "Satlas" # ["Satlas", "ResNetFPN"]
_CN.EPIMASK.COARSE_MATCHING_TYPE = "unmasked" # ["unmasked", "masked"]
_CN.EPIMASK.USE_POS_ENCODING = False #[True, False]
_CN.EPIMASK.RESOLUTION = [8, 2] # options: [[8, 2], [16, 4]]
_CN.EPIMASK.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.EPIMASK.FINE_CONCAT_COARSE_FEAT = True
_CN.EPIMASK.GT_MATCHES_3D_DISTANCE_THR = 0.3 # (meters) euclidean distance threshold on lat, lon, ht to declare a match

# 1. EpiMask-backbone (local feature CNN) config
# resnet-fpn config (loftr backbone)
_CN.EPIMASK.RESNETFPN = CN()
_CN.EPIMASK.RESNETFPN.INITIAL_DIM = 128
_CN.EPIMASK.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3 8_2
# _CN.EPIMASK.RESNETFPN.BLOCK_DIMS = [128, 128, 196, 256]  # s1, s2, s3, s4 16_4

# EpiMask - satlas FPN config
_CN.EPIMASK.SATLAS_FPN = CN()
_CN.EPIMASK.SATLAS_FPN.ENABLED= True          # Set to true to enable the FPN
_CN.EPIMASK.SATLAS_FPN.OUT_CHANNELS = CN()
_CN.EPIMASK.SATLAS_FPN.OUT_CHANNELS.KEYS = ["1/16", "1/8", "1/4", "1/2"]
_CN.EPIMASK.SATLAS_FPN.OUT_CHANNELS.VALUES = [256, 256, 128, 128]      # Number of output channels for the FPN feature maps
_CN.EPIMASK.SATLAS_FPN.NUM_CHANNELS = CN()
_CN.EPIMASK.SATLAS_FPN.NUM_CHANNELS.KEYS = ["1/32", "1/16", "1/8", "1/4"]
_CN.EPIMASK.SATLAS_FPN.NUM_CHANNELS.VALUES = [1024, 512, 256, 128]

# EpiMask - satlas backbone 
_CN.EPIMASK.SATLAS_FPN.PRETRAINED_MODELNAME = "Aerial_SwinB_SI" #options: [@aditya?]
_CN.EPIMASK.SATLAS_FPN.IN_CHANNELS = 1 # number of input channels, options: [1 (gray), 3 (RGB)]
_CN.EPIMASK.SATLAS_FPN.USE_DEFAULT_FPN= False
_CN.EPIMASK.SATLAS_FPN.USE_LORA_FINETUNING= False  # Enable LoRA fine-tuning

# fusion type
_CN.EPIMASK.SATLAS_FPN.FPN_FUSION_TYPE = "concat_conv" # ["concat_conv", "add"]

# LoRA config
_CN.EPIMASK.SATLAS_FPN.LORA = CN()
_CN.EPIMASK.SATLAS_FPN.LORA.CONF_KEYS = ['rank', 'alpha', 'dropout', 'modules']
# _CN.EPIMASK.SATLAS_FPN.LORA.CONF_VALUES = [4, 32, 0.1, ["attn.qkv", "attn.proj","mlp.0", "mlp.3", "cpb_mlp.0", "cpb_mlp.2"]]
_CN.EPIMASK.SATLAS_FPN.LORA.CONF_VALUES = [4, 32, 0.1, ["attn.qkv", "attn.proj","mlp.0", "mlp.3"]]

# 2. EpiMask-coarse module config
_CN.EPIMASK.COARSE = CN()
_CN.EPIMASK.COARSE.CAM_TYPE = 'affine'  # options: ['affine', 'pinhole']
_CN.EPIMASK.COARSE.D_MODEL = 256
_CN.EPIMASK.COARSE.NHEAD = 8
# _CN.EPIMASK.COARSE.D_FFN = 256
_CN.EPIMASK.COARSE.LAYER_NAMES = ['self', 'maskedx'] * 4
_CN.EPIMASK.COARSE.SELF_ATTENTION = 'linear'  # options: ['linear', 'full']

# epipolar distance threshold decay
_CN.EPIMASK.COARSE.EPI_DIST_THR_START = 224
_CN.EPIMASK.COARSE.EPI_DIST_THR_REDUCTION_FACTOR = 0.4


# 3. Coarse-Matching config
_CN.LOFTR.MATCH_COARSE = CN()
_CN.EPIMASK.MATCH_COARSE = CN()
_CN.EPIMASK.MATCH_COARSE.THR = 0.2
_CN.EPIMASK.MATCH_COARSE.BORDER_RM = 2
_CN.EPIMASK.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'  # options: ['dual_softmax, 'sinkhorn']
_CN.LOFTR.MATCH_COARSE.MATCH_TYPE = _CN.EPIMASK.MATCH_COARSE.MATCH_TYPE
_CN.EPIMASK.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.EPIMASK.MATCH_COARSE.SKH_ITERS = 3
_CN.EPIMASK.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
_CN.EPIMASK.MATCH_COARSE.SKH_PREFILTER = False
_CN.EPIMASK.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.EPIMASK.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.EPIMASK.MATCH_COARSE.SPARSE_SPVS = True
_CN.LOFTR.MATCH_COARSE.SPARSE_SPVS = _CN.EPIMASK.MATCH_COARSE.SPARSE_SPVS

# 4. EpiMask-fine module config
_CN.EPIMASK.FINE = CN()
_CN.EPIMASK.FINE.D_MODEL = 128
_CN.EPIMASK.FINE.D_FFN = 128
_CN.EPIMASK.FINE.NHEAD = 8
_CN.EPIMASK.FINE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.EPIMASK.FINE.ATTENTION = 'linear'

# 5. Losses
# -- # coarse-level
_CN.LOFTR.LOSS = CN()
_CN.LOFTR.LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
_CN.LOFTR.LOSS.COARSE_WEIGHT = 1.0
# _CN.EPIMASK.LOSS.SPARSE_SPVS = False
# -- - -- # focal loss (coarse)
_CN.LOFTR.LOSS.FOCAL_ALPHA = 0.25
_CN.LOFTR.LOSS.FOCAL_GAMMA = 2.0
_CN.LOFTR.LOSS.POS_WEIGHT = 1.0
_CN.LOFTR.LOSS.NEG_WEIGHT = 1.0
# _CN.EPIMASK.LOSS.DUAL_SOFTMAX = False  # whether coarse-level use dual-softmax or not.
# use `_CN.EPIMASK.MATCH_COARSE.MATCH_TYPE`

# -- # fine-level
_CN.LOFTR.LOSS.FINE_TYPE = 'l2_with_std'  # ['l2_with_std', 'l2']
_CN.LOFTR.LOSS.FINE_WEIGHT = 1.0
_CN.LOFTR.LOSS.FINE_CORRECT_THR = 1.0  # for filtering valid fine-level gts (some gt matches might fall out of the fine-level window)

# ##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None    # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 640  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8

# SatDepth options
_CN.DATASET.SATDEPTH_IMG_RESIZE = 448  # square image size


##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 6e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning
_CN.TRAINER.GRADIENT_ACCUMULATION = 1

_CN.TRAINER.NUM_EPOCHS_TO_START_EPI_DECAY = 5

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.MAX_PAIRS_TO_PLOT = 8 # number of val/test paris for plotting
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'
_CN.TRAINER.EPIPOLAR_THR = 1.0 # epipolar thr

# geometric metrics and pose solver
# _CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.EPI_ERR_THR = 1.0  
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_RAND_SAMPLE_SIZE = 10
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 1000
_CN.TRAINER.USE_MAGSACPP = False

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()

def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

def get_cfg(config_file:str=''):
    "return a yacs cfgNode object with default values"
    if config_file != '':
        #load config file
        cfg = CN.load_cfg(open(config_file))
        # lower the config
        cfg = lower_config(cfg)
        return cfg
    cn = _CN.clone()
    cfg = lower_config(cn)
    return cfg
