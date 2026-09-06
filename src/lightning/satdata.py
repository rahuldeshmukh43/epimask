import os
import pytorch_lightning as pl
from torch import distributed as dist
from loguru import logger
from torch.utils.data import DataLoader

from epimask.external.satdepth.src.datasets.satdepth import SatDepthLoader
from epimask.src.datasets.satdepth_webdataset import ShardedSatDepthDataset

class SatDataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.rot_aug = args.rot_aug

    def setup(self, stage:str=None):
        # create datasets here: no need to do anything fancy like loftr (ie using _build_concat_dataset)
        # just instantiate the plain old satdepth dataset class for trianing and validation
        assert stage in ['fit', 'test'], "stage must be either fit or test"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set world_size=1 and rank=0)")

        if stage == 'fit':
            self.train_dataset = None #ideally this is where you would instantiate the dataset
            self.val_dataset = None

            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')
        else: # stage == "test"
            NotImplementedError("DataModule not implemented for testing stage")

    def train_dataloader(self):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        if self.rot_aug:
            dataloader = SatDepthLoader(self.args, "train", rotation_augmentation=True).load_data()
        else:
            dataloader = SatDepthLoader(self.args, "train", rotation_augmentation=False).load_data()
        return dataloader

    def val_dataloader(self):
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        dataloader = SatDepthLoader(self.args, "val", rotation_augmentation=False).load_data()
        return dataloader

class SatWebDatasetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        shard_dir: str,
        shards_per_epoch_train: int,
        shard_size_train: int = 500,
        shard_size_val: int = 250,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle_buffer_size: int = 1000,
        world_size: int = 1,
    ):
        super().__init__()
        self.shard_dir = shard_dir
        self.shards_per_epoch_train = shards_per_epoch_train
        self.shard_size_train = shard_size_train
        self.shard_size_val = shard_size_val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.world_size = world_size

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """Called on every GPU in DDP"""
        if stage in (None, "fit"):
            train_dir= os.path.join(self.shard_dir, "train")
            assert os.path.exists(train_dir), f"Train shard directory {train_dir} does not exist"
            self.train_dataset = ShardedSatDepthDataset(
                shard_dir= train_dir,
                shards_per_epoch=self.shards_per_epoch_train,
                shard_size=self.shard_size_train,
                phase="train",
                shuffle_buffer_size=self.shuffle_buffer_size,
                world_size=self.world_size,
            )
            val_dir= os.path.join(self.shard_dir, "val")
            assert os.path.exists(val_dir), f"Val shard directory {val_dir} does not exist"
            self.val_dataset = ShardedSatDepthDataset(
                shard_dir= val_dir,
                shard_size=self.shard_size_val,
                phase="val",
                world_size=self.world_size,
            )
        # if stage in (None, "test"):
        #     self.test_dataset = ShardedSatDepthDataset(
        #         shard_dir=self.shard_dir,
        #         shards_per_epoch=self.shards_per_epoch_test,
        #         phase="test",
        #         shuffle_buffer_size=self.shuffle_buffer_size,
        #     )

    def train_dataloader(self):
        dl = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,  # DataLoader handles batching
            num_workers=self.num_workers,
            pin_memory=False,
        )
        print(f"[DEBUG] train len={len(self.train_dataset)}, batch_size={self.batch_size}, steps/epoch={len(dl)}")
        return dl

    def val_dataloader(self):
        dl = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        print(f"[DEBUG] val len={len(self.val_dataset)}, batch_size={self.batch_size}, steps/epoch={len(dl)}")
        return dl

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=True,
    #     )