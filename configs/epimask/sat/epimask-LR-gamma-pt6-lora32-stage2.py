from epimask.src.config.epimask_default import _CN as cfg

cfg.EPIMASK.RESOLUTION = [8, 2]
cfg.EPIMASK.COARSE_MATCHING_TYPE = "masked" # ["unmasked", "masked"]
cfg.EPIMASK.USE_POS_ENCODING = True
cfg.EPIMASK.SATLAS_FPN.FPN_FUSION_TYPE = "concat_conv" # ["concat_conv", "add"]

cfg.EPIMASK.GT_MATCHES_3D_DISTANCE_THR = 0.3 # meters

cfg.EPIMASK.MATCH_COARSE.MATCH_TYPE = 'dual_softmax' # options: ['dual_softmax, 'sinkhorn']
cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = cfg.EPIMASK.MATCH_COARSE.MATCH_TYPE
cfg.EPIMASK.MATCH_COARSE.SPARSE_SPVS = False
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = cfg.EPIMASK.MATCH_COARSE.SPARSE_SPVS

cfg.TRAINER.CANONICAL_LR = 8e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.EPIMASK.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3

cfg.TRAINER.GRADIENT_ACCUMULATION = 8

cfg.TRAINER.NUM_EPOCHS_TO_START_EPI_DECAY = 5

# epipolar distance threshold decay
# gamma (EPI_DIST_THR_REDUCTION_FACTOR) below is 0.6 to match the released
# EpiMask-LR gamma=0.6 checkpoint (model_weights/epimask-LR-gamma-pt6-lora32-stage2/);
# set it to 0.4 to instead train/reproduce the gamma=0.4 variant.
cfg.EPIMASK.COARSE.EPI_DIST_THR_START = 224
cfg.EPIMASK.COARSE.EPI_DIST_THR_REDUCTION_FACTOR = 0.6

# lora configs
cfg.EPIMASK.SATLAS_FPN.USE_LORA_FINETUNING = True  # Enable LoRA fine-tuning

# lora 32
cfg.EPIMASK.SATLAS_FPN.LORA.CONF_KEYS = ['rank', 'alpha', 'dropout', 'modules','rank_pattern','alpha_pattern']
cfg.EPIMASK.SATLAS_FPN.LORA.CONF_VALUES = [32, 16, 0.1, ["attn.qkv", "attn.proj","mlp.0", "mlp.3", "reduction"],{"attn.qkv": 96},{"attn.qkv": 48}]
