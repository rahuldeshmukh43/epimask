import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedDataset


class SetEpochCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        ds = trainer.train_dataloader.dataset

        if isinstance(ds, CombinedDataset):
            d = ds.datasets
            if hasattr(d, "set_epoch"):
                print(f"SetEpochCallback:  Setting epoch to {epoch} for sub-dataset {d}")
                d.set_epoch(epoch)
            else:
                print(f"SetEpochCallback:  Sub-dataset {d} has no set_epoch method")
        else:
            if hasattr(ds, "set_epoch"):
                print(f"SetEpochCallback:  Setting epoch to {epoch} for dataset {ds}")
                ds.set_epoch(epoch)
            else:
                print(f"SetEpochCallback:  Dataset {ds} has no set_epoch method")
        