import os
import re
import json
import struct
import io

import numpy as np
import configargparse
import webdataset as wds

from epimask.external.satdepth.src.datasets.satdepth import SatDepthDataset, _read_pairs
from epimask.src.utils.testing_utils import make_dsm_gridpts
from epimask.external.satdepth.src.utils.useful_methods import get_basename

# float → bytes
def float_to_bytes(x: float) -> bytes:
    return struct.pack("f", x)

# tuple of strings → bytes
def tuple_str_to_bytes(t: tuple) -> bytes:
    return "|".join(t).encode("utf-8")

def np_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    # allow_pickle=False for safety/reproducibility
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()

def write_sharded_webdataset(
    samples_iter,
    shard_top_dir,
    shard_pattern="satdepth-{000000..009999}.tar",
    sharded_dataset_total_num_samples:int = 1e6,
    maxcount:int=10000,
    maxsize:int=1e9,
    ):
    """
    Writes samples yielded by samples_iter to shards. Samples are dicts as
    returned by SatDepthDataset.__getitem__, which vary slightly by how the
    dataset was constructed: randomly-sampled patches (img_pair=None,
    grid_pts=None; used for --phase train/val) carry rotation_aug_angle /
    relative_track_angle, while grid-tiled whole-image patches (img_pair set,
    grid_pts set; used for --phase test) carry img0_extents / img1_extents
    instead. Both are written when present so this one writer covers both.
    Returns the number of samples actually written (None samples are skipped).
    """
    shard_pattern = os.path.join(shard_top_dir, shard_pattern)
    count = 0
    itr_count = 0
    with wds.ShardWriter(shard_pattern, maxcount=maxcount, maxsize=maxsize) as sink:
        while itr_count < sharded_dataset_total_num_samples:
            s = next(samples_iter)
            if s is None:
                itr_count += 1
                continue

            key = f"{count:06d}"

            # images → numpy bytes
            img0 = np_to_bytes(s["image0"].squeeze(0).numpy())
            img1 = np_to_bytes(s["image1"].squeeze(0).numpy())

            # cameras
            affine_cam0 = np_to_bytes(s["affine_cam0"])
            affine_cam1 = np_to_bytes(s["affine_cam1"])

            # scalars → float bytes
            lat0 = np_to_bytes(s["lat0"])
            lon0 = np_to_bytes(s["lon0"])
            ht0 = np_to_bytes(s["ht0"])
            lat1 = np_to_bytes(s["lat1"])
            lon1 = np_to_bytes(s["lon1"])
            ht1 = np_to_bytes(s["ht1"])

            intersection_angle = float_to_bytes(s["intersection_angle"])

            # F_gt → numpy bytes
            F_gt = np_to_bytes(s["F_gt"])

            # strings / tuple → bytes
            dataset_name = s["dataset_name"].encode("utf-8")
            pair_names = tuple_str_to_bytes(s["pair_names"])

            # assemble sample
            sample = {
                "__key__": key,
                "image0": img0,
                "image1": img1,
                "affine_cam0": affine_cam0,
                "lat0": lat0,
                "lon0": lon0,
                "ht0": ht0,
                "affine_cam1": affine_cam1,
                "lat1": lat1,
                "lon1": lon1,
                "ht1": ht1,
                "F_gt": F_gt,
                "intersection_angle": intersection_angle,
                "dataset_name": dataset_name,
                "pair_names": pair_names,
            }

            # present for --phase train/val (randomly-sampled patches)
            if "rotation_aug_angle" in s:
                sample["rotation_aug_angle"] = float_to_bytes(s["rotation_aug_angle"])
                sample["relative_track_angle"] = float_to_bytes(s["relative_track_angle"])

            # present for --phase test (grid-tiled whole-image patches)
            if "img0_extents" in s:
                sample["img0_extents"] = np_to_bytes(s["img0_extents"])
                sample["img1_extents"] = np_to_bytes(s["img1_extents"])

            sink.write(sample)
            count += 1
            itr_count += 1
    return count

def cycle_dataset(dataset, num_samples=None):
    i = 0
    n = len(dataset)
    while num_samples is None or i < num_samples:
        if i % n == 0 and i > 0:
            print(f"Recycling dataset after {i} samples")
        yield dataset[i % n]  # cycles if needed
        i += 1


def parse_args():
    """Parses dataset-construction args used by SatDepthDataset plus shard-writing options."""
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument("--config", is_config_file=True, help="config file path (see src/datasets/sharded_dataset_configs/)")

    parser.add_argument("--phase", type=str, required=True, choices=["train", "val", "test"],
                         help="train/val: randomly-sampled patches oversampled to a target sample count. "
                              "test: whole-image patches tiled over a DSM grid, one shard set per pair "
                              "(matches what test_epimask_sharded.py expects).")
    parser.add_argument("--train_pairlist", type=str, default=None, help="path to pair list file for training (required if --phase train)")
    parser.add_argument("--val_pairlist", type=str, default=None, help="path to pair list file for validation (required if --phase val)")
    parser.add_argument("--test_pairlist", type=str, default=None, help="path to pair list file for the AOI being sharded (required if --phase test)")

    # dataset construction options
    parser.add_argument("--img_patch_size", type=int, default=448, help="patch size extracted from original image")
    parser.add_argument("--train_img_size", type=int, default=448, help="size of patch used for training the network")
    parser.add_argument("--nodata_value", type=float, default=-9999, help="no data value for lat/lon/ht maps")
    parser.add_argument("--dsm_shrink_buffer", type=int, default=250, help="single side dsm shrinking buffer")
    parser.add_argument("--rot_aug", action="store_true", help="flag for rotation augmentation while sharding (train/val only; unused for --phase test)")
    parser.add_argument("--funda_method", type=str, default="cameras", help="cameras/matches for calculating affine fundamental matrix")
    parser.add_argument("--num_pts", type=int, default=800, help="num of points to be extracted in each pair")
    parser.add_argument("--num_pts_retained", type=int, default=50, help="num of points to be retained for computing fundamental matrix")
    parser.add_argument("--kp_mode", type=str, default="mixed", help="sift/random/mixed")
    parser.add_argument("--pct_sift", type=float, default=0.9, help="percentage of sift points when mode is mixed")
    parser.add_argument("--kp_distance_thresh", type=float, default=0.25, help="3d distance threshold in meters to ascertain if a true match")

    # shard-writing options
    parser.add_argument("--shard_dir", type=str, required=True,
                         help="output dir for shards, relative to the repo root. For --phase train/val, a "
                              "'<phase>/' subfolder is created under it to match SatWebDatasetDataModule's "
                              "expected layout. For --phase test, shards are written directly under it "
                              "(one shard set per AOI pair), matching the shard_dir expected by test_epimask_sharded.py")
    parser.add_argument("--sharded_dataset_total_num_samples", type=int, default=None,
                         help="total number of samples to write (the source pairlist is cycled if smaller). "
                              "Required for --phase train/val; ignored for --phase test, where each pair's "
                              "full DSM grid is written instead.")
    parser.add_argument("--max_records_per_shard", type=int, default=500, help="number of samples per shard (.tar) file (--phase train/val only)")
    parser.add_argument("--maxsize", type=float, default=10e9, help="max shard file size in bytes")

    return parser.parse_args()


if __name__ == "__main__":
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    args = parse_args()

    if args.phase == "train" and not args.train_pairlist:
        raise ValueError("--train_pairlist is required when --phase train")
    if args.phase == "val" and not args.val_pairlist:
        raise ValueError("--val_pairlist is required when --phase val")
    if args.phase == "test" and not args.test_pairlist:
        raise ValueError("--test_pairlist is required when --phase test")
    if args.phase in ("train", "val") and not args.sharded_dataset_total_num_samples:
        raise ValueError("--sharded_dataset_total_num_samples is required when --phase train/val")

    # By default, shards are written under data/satdepth/webdataset/ inside the repo.
    # This directory is not committed (too large) -- either let this script populate it,
    # or symlink it to wherever your shard storage actually lives.
    if args.phase in ("train", "val"):
        shard_top_dir = os.path.join(PROJECT_DIR, args.shard_dir, args.phase)
        print("Writing sharded dataset to %s" % shard_top_dir)
        os.makedirs(shard_top_dir, exist_ok=True)

        dataset = SatDepthDataset(args,
                        args.phase,
                        rotation_augmentation=args.rot_aug,
                        img_pair=None,
                        grid_pts=None)

        dataset_iter = cycle_dataset(dataset, num_samples=args.sharded_dataset_total_num_samples)

        shard_pattern = "satdepth-%dsamples-shard-" % args.max_records_per_shard
        shard_pattern += "%06d.tar"
        write_sharded_webdataset(dataset_iter,
                                shard_top_dir,
                                shard_pattern=shard_pattern,
                                sharded_dataset_total_num_samples=args.sharded_dataset_total_num_samples,
                                maxcount=args.max_records_per_shard,
                                maxsize=args.maxsize)

    else:  # phase == "test"
        shard_top_dir = os.path.join(PROJECT_DIR, args.shard_dir)
        print("Writing sharded dataset to %s" % shard_top_dir)
        os.makedirs(shard_top_dir, exist_ok=True)

        test_pairs = _read_pairs(args.test_pairlist)
        test_pairs_num_samples = {}

        for test_pair in test_pairs:
            try:
                img0_ds, img1_ds, intersection_angle, relative_track_angle = test_pair
            except ValueError:
                img0_ds, img1_ds, intersection_angle = test_pair

            # compute grid on dsm
            grid_pts = make_dsm_gridpts(img0_ds.dsm_path,
                                    args.dsm_shrink_buffer,
                                    args.img_patch_size)

            dataset = SatDepthDataset(args,
                            "test",
                            rotation_augmentation=False,
                            img_pair=test_pair,
                            grid_pts=grid_pts)
            dataset_len = len(dataset)

            dataset_iter = iter(dataset)

            img0_base = get_basename(test_pair[0].img_path)
            img1_base = get_basename(test_pair[1].img_path)
            aoi_name = re.search("aoi_rect_piece_[0-9]*", test_pair[0].img_path)
            aoi_name = aoi_name.group(0)

            shard_pattern = "%s-%s_and_%s-shard-" % (aoi_name, img0_base, img1_base)
            shard_pattern += "%06d.tar"
            this_pair_num_samples = write_sharded_webdataset(dataset_iter,
                                    shard_top_dir,
                                    shard_pattern=shard_pattern,
                                    sharded_dataset_total_num_samples=dataset_len,
                                    maxcount=dataset_len,
                                    maxsize=args.maxsize)
            test_pairs_num_samples["%s-%s_and_%s" % (aoi_name, img0_base, img1_base)] = this_pair_num_samples

            print(f"Wrote {this_pair_num_samples} samples for pair {img0_base} and {img1_base}")

        # write num_samples for the aoi as a json (consumed by test_epimask_sharded.py)
        out_file = os.path.join(shard_top_dir, "all_pairs_num_samples.json")
        with open(out_file, 'w') as f:
            json.dump(test_pairs_num_samples, f, indent=4)
        print(f"Wrote num_samples for all pairs to {out_file}")
