import os
import math
import configargparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from epimask.external.LoFTR.src.utils.misc import get_rank_zero_only_logger, setup_gpus
from epimask.external.LoFTR.src.utils.profiler import build_profiler

from epimask.src.config.epimask_default import get_cfg_defaults
from epimask.src.lightning.lightning_epimask import PL_EpiMask
from epimask.src.lightning.satdata import SatDataModule, SatWebDatasetDataModule
from epimask.src.utils.callbacks import SetEpochCallback

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument("--config", is_config_file=True, help="config file path")

    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=False, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained EpiMask')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')
    parser.add_argument("--logdir", type=str, default="./logs/", help="dir of tensorboard logs and model checkpoints")

    # sat related args
    parser.add_argument("--train_pairlist", type=str,  help="path to pair list file for training")
    parser.add_argument("--val_pairlist", type=str, help="path to pair list file for validation")
    parser.add_argument("--multi_gpu", action="store_true", help="flag for muti-gpu training")

    # dataset type - ["default", "sharded"]
    parser.add_argument("--dataset_type", type=str, default="default", help="dataset type - [default, sharded]")
    
    #default DATASET options:
    parser.add_argument("--img_patch_size", type=int, default=400, help="patch size extracted from original image")
    parser.add_argument("--train_img_size", type=int, default=400, help="size of patch used for training the network")
    parser.add_argument("--nodata_value", type=float, default=-9999, help="no data value for lat/lon/ht maps")
    parser.add_argument("--dsm_shrink_buffer", type=int, default=250, help="single side dsm shrinking buffer")
    parser.add_argument("--rot_aug", action="store_true", help="flag for rotation augmentation during training")
    parser.add_argument("--funda_method", type=str, default="cameras", help="cameras/matches for calculating affine fundamental matrix" )

    parser.add_argument("--num_pts", type=int, default=800, help="num of points to be extracted in each pair")
    parser.add_argument("--num_pts_retained", type=int, default=50, help="num of points to be retained for computing fundamental matrix")
    parser.add_argument("--kp_mode", type=str, default="mixed", help="sift/random/mixed")
    parser.add_argument("--pct_sift", type=float, default=0.9, help="percentage of sift points when mode is mixed")
    parser.add_argument("--kp_distance_thresh", type=float, default=0.25, help="3d distance threshold in meters to ascertain if a true match")

    # sharded dataset related args
    parser.add_argument("--shard_dir", type=str, default=None, help="path to sharded dataset directory")
    parser.add_argument("--shards_per_epoch_train", type=int, default=20, help="number of shards to be drawn in each training epoch")
    parser.add_argument("--shuffle_buffer_size", type=int, default=1000, help="shuffle buffer size for sharded dataset")
    # shard sizes
    parser.add_argument("--shard_size_train", type=int, default=500, help="number of samples in each training shard")
    parser.add_argument("--shard_size_val", type=int, default=250, help="number of samples in each validation shard")

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

@rank_zero_only
def save_config_and_args(args, config, save_dir):
    # write the final config and args to save_dir
    with open(os.path.join(save_dir, 'train_config.yaml'), 'w') as f:
        f.write(config.dump())
    with open(os.path.join(save_dir, 'train_args.yaml'), 'w') as f:
        f.write(pprint.pformat(vars(args)))
    print(f"Config and args saved to {save_dir}")
    return

def main():
    # parse arguments
    args = parse_args()
    save_dir = args.logdir
    rank_zero_only(os.makedirs)(save_dir, exist_ok=True)
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    
    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)

    # write the final config and args to save_dir
    save_config_and_args(args, config, save_dir)
    
    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_EpiMask(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"EpiMask LightningModule initialized!")
    
    # lightning data
    if args.dataset_type == "default":
        data_module = SatDataModule(args, config)
    elif args.dataset_type == "sharded":
        data_module = SatWebDatasetDataModule(
            shard_dir=args.shard_dir,
            shards_per_epoch_train=args.shards_per_epoch_train,
            shard_size_train=args.shard_size_train,
            shard_size_val=args.shard_size_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_buffer_size=args.shuffle_buffer_size,
            world_size=config.TRAINER.WORLD_SIZE
        )
        # print len of train and val datasets
        data_module.setup('fit')
        loguru_logger.info(f"len(train_dataset): {len(data_module.train_dataset)}")
        loguru_logger.info(f"len(val_dataset): {len(data_module.val_dataset)}")

    loguru_logger.info(f"EpiMask-Sat DataModule initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir=os.path.join(save_dir, 'logs/tb_logs'), name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    
    # Callbacks
    ckpt_callback = ModelCheckpoint(monitor='val_loss', # monitor='auc@10',
                                    verbose=True,
                                    # save_top_k=5,
                                    mode='min', #mode='max',
                                    save_last=True,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{val_loss:.4f}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    
    if args.dataset_type == "sharded":
        set_epoch_callback = SetEpochCallback()
        callbacks.append(set_epoch_callback)

    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)
    
    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(find_unused_parameters=True,
                          num_nodes=args.num_nodes,
                          sync_batchnorm=config.TRAINER.WORLD_SIZE > 0),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=False,  # avoid repeated samples!
        weights_summary='full',
        profiler=profiler,
        )
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
