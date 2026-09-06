import os
import webdataset as wds
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset

import io
import struct

def bytes_to_float(b: bytes) -> float:
    return struct.unpack("f", b)[0]

def bytes_to_tuple_str(b: bytes) -> tuple:
    return tuple(b.decode("utf-8").split("|"))

def bytes_to_numpy(b: bytes) -> np.ndarray:
    buf = io.BytesIO(b)
    return np.load(buf, allow_pickle=False)

def post_process(sample_tuple):
    (img0_bytes, img1_bytes,
    affine_cam0_bytes, affine_cam1_bytes,
    lat0_bytes, lon0_bytes, ht0_bytes,
    lat1_bytes, lon1_bytes, ht1_bytes,
    F_bytes,
    intersection_angle_bytes,
    dataset_name_bytes,
    pair_names_bytes,
    rotation_aug_angle_bytes,
    relative_track_angle_bytes) = sample_tuple

    img0 = bytes_to_numpy(img0_bytes)
    img1 = bytes_to_numpy(img1_bytes)
    affine_cam0 = bytes_to_numpy(affine_cam0_bytes)
    affine_cam1 = bytes_to_numpy(affine_cam1_bytes)
    lat0, lon0, ht0 = bytes_to_numpy(lat0_bytes), bytes_to_numpy(lon0_bytes), bytes_to_numpy(ht0_bytes)
    lat1, lon1, ht1 = bytes_to_numpy(lat1_bytes), bytes_to_numpy(lon1_bytes), bytes_to_numpy(ht1_bytes)
    F_gt = bytes_to_numpy(F_bytes)
    intersection_angle = bytes_to_float(intersection_angle_bytes)
    rotation_aug_angle = bytes_to_float(rotation_aug_angle_bytes)
    relative_track_angle = bytes_to_float(relative_track_angle_bytes)
    dataset_name = dataset_name_bytes.decode("utf-8")
    pair_names = bytes_to_tuple_str(pair_names_bytes)

    # convert to torch tensors
    img0 = torch.from_numpy(img0).unsqueeze(0)  # (1, H, W)
    img1 = torch.from_numpy(img1).unsqueeze(0)  # (1, H, W)

    return {
        "image0": img0,
        "image1": img1,
        "affine_cam0": affine_cam0,
        "affine_cam1": affine_cam1,
        "lat0": lat0,
        "lon0": lon0,
        "ht0": ht0,
        "lat1": lat1,
        "lon1": lon1,
        "ht1": ht1,
        "F_gt": F_gt,
        "intersection_angle": intersection_angle,
        "rotation_aug_angle": rotation_aug_angle,
        "relative_track_angle": relative_track_angle,
        "dataset_name": dataset_name,
        "pair_names": pair_names,
    }

class ShardedSatDepthDataset(IterableDataset):
    def __init__(
        self,
        shard_dir,
        phase:str,
        shards_per_epoch: int = None,
        shard_size:int = 500,
        shuffle_buffer_size=1000,
        world_size: int = 1,
    ):
        """
        shard_dir: directory containing shards
        phase: "train", "val", or "test" 
        shards_per_epoch: how many shards to draw each epoch (for training only)
        shuffle_buffer_size: shuffle buffer size (for training only)
        """
        self.shard_dir = shard_dir
        assert os.path.exists(shard_dir), f"Shard directory {shard_dir} does not exist"
        self.shards_per_epoch = shards_per_epoch
        self.shard_size = shard_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.world_size = world_size

        assert phase in ["train", "val", "test"]
        self.phase = phase
        self.current_epoch = 0
        self.prev_shards = set()

        # assume shards are named *.tar
        self.all_shards = [os.path.join(shard_dir, name) for name in os.listdir(shard_dir) if name.endswith('.tar')]
        self.num_shards = len(self.all_shards)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _sample_shards(self):
        rng = random.Random(self.current_epoch)

        # available = all shards except those from previous epoch
        available_ids = list(set(range(self.num_shards)) - self.prev_shards)

        # if too few left, reset to full pool
        if len(available_ids) < self.shards_per_epoch:
            available_ids = list(range(self.num_shards))

        shard_ids = rng.sample(available_ids, self.shards_per_epoch)

        # update prev_shards
        self.prev_shards = set(shard_ids)

        # return actual file paths
        urls = [self.all_shards[sid] for sid in shard_ids]

        print(f"\nEpoch {self.current_epoch}: using shard ids:- {shard_ids}")
        # print(*shard_ids, sep=" ")
        return urls

    def __len__(self):
        if self.phase == "train":
            return self.shards_per_epoch * self.shard_size // self.world_size
        else:
            return self.num_shards * self.shard_size // self.world_size
        

    def __iter__(self):
        if self.phase == "train":
            urls = self._sample_shards()
        else:
            urls = self.all_shards

        dataset = (
            wds.WebDataset(urls, 
                        shardshuffle=False, 
                        nodesplitter=wds.split_by_node,
                        workersplitter=wds.split_by_worker, 
                        empty_check=True)
            .shuffle(self.shuffle_buffer_size if self.phase == "train" else 0)
            .to_tuple(
                "image0", "image1",
                "affine_cam0", "affine_cam1",
                "lat0", "lon0", "ht0",
                "lat1", "lon1", "ht1",
                "f_gt",
                "intersection_angle",
                "dataset_name",
                "pair_names",
                "rotation_aug_angle",
                "relative_track_angle",
            )
            .map(post_process)
        )
        return iter(dataset)
    
def post_process_test(sample_tuple):
    (img0_bytes, img1_bytes,
    affine_cam0_bytes, affine_cam1_bytes,
    lat0_bytes, lon0_bytes, ht0_bytes,
    img0_extents,
    lat1_bytes, lon1_bytes, ht1_bytes,
    img1_extents,
    F_bytes,
    intersection_angle_bytes,
    dataset_name_bytes,
    pair_names_bytes,
    ) = sample_tuple

    img0 = bytes_to_numpy(img0_bytes)
    img1 = bytes_to_numpy(img1_bytes)
    affine_cam0 = bytes_to_numpy(affine_cam0_bytes)
    affine_cam1 = bytes_to_numpy(affine_cam1_bytes)
    lat0, lon0, ht0 = bytes_to_numpy(lat0_bytes), bytes_to_numpy(lon0_bytes), bytes_to_numpy(ht0_bytes)
    img0_extents = bytes_to_numpy(img0_extents)
    lat1, lon1, ht1 = bytes_to_numpy(lat1_bytes), bytes_to_numpy(lon1_bytes), bytes_to_numpy(ht1_bytes)
    img1_extents = bytes_to_numpy(img1_extents)
    F_gt = bytes_to_numpy(F_bytes)
    intersection_angle = bytes_to_float(intersection_angle_bytes)

    dataset_name = dataset_name_bytes.decode("utf-8")
    pair_names = bytes_to_tuple_str(pair_names_bytes)

    # convert to torch tensors
    img0 = torch.from_numpy(img0).unsqueeze(0)  # (1, H, W)
    img1 = torch.from_numpy(img1).unsqueeze(0)  # (1, H, W)

    return {
        "image0": img0,
        "image1": img1,
        "affine_cam0": affine_cam0,
        "affine_cam1": affine_cam1,
        "lat0": lat0,
        "lon0": lon0,
        "ht0": ht0,
        "img0_extents": img0_extents,
        "lat1": lat1,
        "lon1": lon1,
        "ht1": ht1,
        "img1_extents": img1_extents,
        "F_gt": F_gt,
        "intersection_angle": intersection_angle,
        "dataset_name": dataset_name,
        "pair_names": pair_names,
    }
    
class ShardedSatDepthDataset_Test_Whole_Image(ShardedSatDepthDataset):
    def __iter__(self):
        assert self.phase == "val"
        urls = self.all_shards

        dataset = (
            wds.WebDataset(urls, 
                        shardshuffle=False, 
                        nodesplitter=wds.split_by_node,
                        workersplitter=wds.split_by_worker, 
                        empty_check=True)
            .shuffle(self.shuffle_buffer_size if self.phase == "train" else 0)
            .to_tuple(
                "image0", "image1",
                "affine_cam0", "affine_cam1",
                "lat0", "lon0", "ht0",
                "img0_extents",
                "lat1", "lon1", "ht1",
                "img1_extents",
                "f_gt",
                "intersection_angle",
                "dataset_name",
                "pair_names",
            )
            .map(post_process_test)
        )
        return iter(dataset)

if __name__ == "__main__":
    # test the dataset
    import os
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    shard_dir = os.path.join(PROJECT_DIR, "data/satdepth/webdataset/satdepth-hw448-128k-rot_aug/train")
    shards_per_epoch = 2
    train_dataset = ShardedSatDepthDataset(shard_dir=shard_dir,
                                        shards_per_epoch=shards_per_epoch,
                                        phase="train")
    import matplotlib.pyplot as plt

    for i, s in enumerate(train_dataset):
        if i >= 1:
            break
        print('-'*100)
        print(s["image0"].shape, s["image1"].shape, s["F_gt"].shape)
        print(s["affine_cam0"].shape, s["affine_cam1"].shape)
        print(s["dataset_name"], s["pair_names"])
        print(s["intersection_angle"], s["rotation_aug_angle"], s["relative_track_angle"])
        print(s["lat0"].shape, s["lat1"].shape)
        print(s["lon0"].dtype)
        # plot image and save as png
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(s["image0"].squeeze().numpy(), cmap='gray')
        axs[1].imshow(s["image1"].squeeze().numpy(), cmap='gray')
        plt.savefig(f"sample_{i}.png")
        plt.close(fig)
