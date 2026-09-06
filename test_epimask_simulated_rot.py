import os, re, sys
import configargparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from epimask.src.model.epimask import EpiMask
from epimask.src.config.epimask_default import get_cfg_defaults, lower_config, get_cfg
from epimask.external.satdepth.src.datasets.satdepth import SatDepthLoader, _read_pairs
from epimask.src.datasets.satdepth_webdataset import ShardedSatDepthDataset
import epimask.external.satdepth.src.model.utils.sat_supervision as sat_supervision
import epimask.src.utils.sat_metrics as sat_metrics
import epimask.external.satdepth.src.utils.satdepth_utils as satdepth_utils
from epimask.external.satdepth.src.utils.useful_methods import setup_logger, get_basename, timeit, pkl_read, pkl_write

from epimask.src.utils.testing_utils import AverageMeter, make_dsm_gridpts, plot_matches

# set random seed -- for reproducibility
SEED = 2024
torch.manual_seed(SEED)
np.random.seed(SEED)

def run(args,
        model_config,
        epi_thrs,
        out_folder:str,
        logger):
    # define model
    model = EpiMask(config=model_config['epimask'])
    model.apply_epipolar_dist_thr_decay()
    
    
    logger.info("Loading checkpoint from %s"%(args.ckpt_path))
    state_dict = torch.load(args.ckpt_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, strict=True)

    device = "cuda:%d" % (args.device) if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # read pair list file
    test_pairs = _read_pairs(args.test_pairlist)
    Npairs = len(test_pairs)
    # set args 
    args.train_pairlist = args.test_pairlist

    p_patches = [[] for _ in range(len(epi_thrs))]
    r_patches = [[] for _ in range(len(epi_thrs))]
    n_correct_patches = [[] for _ in range(len(epi_thrs))]
    n_gt_matches_patches = [[] for _ in range(len(epi_thrs))]
    n_detected_patches = [[] for _ in range(len(epi_thrs))]
    phi_errs_patches = []
    theta_errs_patches = []
    s_errs_patches = []
    intersection_angles_per_image = []
    rel_track_angles_per_image = []

    # instantiate dataloader and run one epoch ie one pass over the dataset
    if args.dataset_type == "default":
        test_dataloader = SatDepthLoader(args, 
                                        "train", # we use train config to allow drawing of random patch
                                        rotation_augmentation=True,
                                        use_seed=True).load_data()
    elif args.dataset_type == "sharded":
        test_dataset = ShardedSatDepthDataset(
                shard_dir= args.shard_dir,
                shard_size=args.shard_size,
                phase="val",
                world_size=1
            )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size= args.batch_size,
            num_workers= args.num_workers,
            pin_memory=False,
        )
    
    @timeit
    def model_pass(model, batch):
        with torch.no_grad():
            model(batch)
        return

    for i_epoch in range(args.num_test_epochs):
        logger.info("Epoch: %d"%(i_epoch))
        for i_batch, data  in enumerate(test_dataloader):
            if not isinstance(data, dict):
                    logger.info("[Testing] received empty batch, continue to next batch (batch: %d)"%(i_batch))
                    continue
            batch_size = len(data["dataset_name"])
            logger.info("[Testing] Batch: %d, Batch Size: %d"%(i_batch, batch_size))

            # send stuff to gpu
            data["image0"] = data["image0"].to(device)
            data["image1"] = data["image1"].to(device)

            # lat lon ht
            data["lat0"] = data["lat0"].to(device)
            data["lon0"] = data["lon0"].to(device)
            data["ht0"] = data["ht0"].to(device)

            data["lat1"] = data["lat1"].to(device)
            data["lon1"] = data["lon1"].to(device)
            data["ht1"] = data["ht1"].to(device)

            # affine camera
            data["affine_cam0"] = data["affine_cam0"].to(device)
            data["affine_cam1"] = data["affine_cam1"].to(device)

            # Fundamental matrix
            data["F_gt"] = data["F_gt"].to(device)

            # intersection angles
            intersection_angles = data["intersection_angle"] #tensor [B] in degree

            # rotation angles
            # rotation_aug_angles = data["rotation_aug_angle"] #[B] in degree
            relative_track_angles = data["relative_track_angle"] #tensor [B] in degree

            # append angles to global list
            intersection_angles_per_image.extend(intersection_angles.tolist())
            rel_track_angles_per_image.extend(relative_track_angles.tolist())

            # get matches for each patch
            model_pass(model, data)

            # do supervision, compute metrics, compute precision recall
            sat_supervision.coarse_supervision(data, model_config["epimask"])
            sat_metrics.compute_distance_errors(data)
            sat_metrics.compute_pose_errors(data)

            # get pose errors        
            _phi_errs, _theta_errs, _s_errs = data["phi_errs"], data["theta_errs"], data["s_errs"]

            for i_thr, thr in enumerate(epi_thrs):
                (_precision, 
                _recall,
                _count,
                _n_correct,
                _n_gt_matches,
                _n_detected, 
                _epi_errs) = sat_metrics.compute_scores(data,
                                                        epipolar_thr=thr,
                                                        nMatchesKeep=args.nMatchesKeep)

                # store for patches
                p_patches[i_thr].extend(_precision)
                r_patches[i_thr].extend(_recall)
                n_correct_patches[i_thr].extend(_n_correct)
                n_gt_matches_patches[i_thr].extend(_n_gt_matches)
                n_detected_patches[i_thr].extend(_n_detected)

            phi_errs_patches.extend(_phi_errs)
            theta_errs_patches.extend(_theta_errs)
            s_errs_patches.extend(_s_errs)
    return (
        p_patches, 
        r_patches,
        n_correct_patches, 
        n_gt_matches_patches, 
        n_detected_patches,
        phi_errs_patches,
        theta_errs_patches, 
        s_errs_patches,
        intersection_angles_per_image,
        rel_track_angles_per_image)

if __name__ == "__main__":
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument("--config", is_config_file=True, help="config file path")

    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="specific checkpoint path to load the model from, "
                            "if not specified, automatically reload from most recent checkpoints")
    parser.add_argument("--model_cfg_path", type=str, default="",
                        help="path to the train_config.yaml matching ckpt_path, e.g. "
                             "training_experiments/<exp_name>/train_config.yaml for a checkpoint "
                             "produced by scripts/epimask/train.sh, or the train_config.yaml bundled "
                             "with a released checkpoint under model_weights/<name>/. train_args.yaml "
                             "is expected alongside it in the same directory.")

    parser.add_argument("--outdir", type=str, default="./out/", help="dir to write test outputs (matches, plots, logs, summary.pkl) to")
    parser.add_argument("--log_name", type=str, default="testing", help="base file name for logging")
    parser.add_argument("--testing_foldername", type=str, default="testing_set", help="folder name for testing set")

    # sat related
    parser.add_argument("--test_pairlist", type=str, help="path to pair list file for testing")

    # GENERAL OPTIONS: experiment name, mode
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--multi_gpu", action="store_true", help="flag for muti-gpu training")
    
    # dataloader options
    parser.add_argument("--img_patch_size", type=int, default=448, help="patch size extracted from original image")
    parser.add_argument("--train_img_size", type=int, default=448, help="size of patch used for training the network")
    parser.add_argument("--nodata_value", type=float, default=-9999, help="no data value for lat/lon/ht maps")
    parser.add_argument("--dsm_shrink_buffer", type=int, default=250, help="single side dsm shrinking buffer")
    parser.add_argument("--rot_aug", action="store_true", help="flag for rotation augmentation during training")
    parser.add_argument("--funda_method", type=str, default="cameras",
                        help="cameras/matches for calculating affine fundamental matrix")

    parser.add_argument("--num_pts", type=int, default=800, help="num of points to be extracted in each pair")
    parser.add_argument("--num_pts_retained", type=int, default=50, help="num of points to be retained for computing fundamental matrix")
    parser.add_argument("--kp_mode", type=str, default="mixed", help="sift/random/mixed")
    parser.add_argument("--pct_sift", type=float, default=0.9, help="percentage of sift points when mode is mixed")
    parser.add_argument("--kp_distance_thresh", type=float, default=0.25, help="3d distance threshold in meters to ascertain if a true match")

    # matcher options
    parser.add_argument("--match_thr", type=float,default=0.5, help="model coarse matching threshold")
    parser.add_argument("--nMatchesKeep", type=int, default=-1, help="number of matches to keep per data sample (ie per patch size). Top N score matches will be retained")

    parser.add_argument("--debug", action="store_true", help="flag for debugging code: only one image will be processed")

    # plot options
    parser.add_argument("--do_match_plot", action="store_true", help="flag for doing match plot")
    # num_test_epochs
    parser.add_argument("--num_test_epochs", type=int, default=1, help="number of epochs to run testing")

    parser.add_argument("--dataset_type", type=str, default="default", help="dataset type [default, sharded]")
    parser.add_argument("--shard_dir", type=str, default="", help="path to shards")
    parser.add_argument("--shard_size", type=int, default=250, help="size of individual shard")

    
    # parse args
    args = parser.parse_args()

    # load model config (path is explicit -- does not need to be under outdir, e.g. for
    # released checkpoints whose train_config.yaml lives under model_weights/<name>/)
    exp_model_config_file = args.model_cfg_path
    assert os.path.exists(exp_model_config_file), "model config file does not exist: %s" % (exp_model_config_file)

    out_folder = os.path.join(args.outdir, args.testing_foldername)
    out_folder += "_match_thr_%0.1f_k%d_hw%d"%(args.match_thr, args.nMatchesKeep, args.img_patch_size)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    log_file = os.path.join(out_folder, args.log_name + '.log')
    logger = setup_logger(args.log_name, log_file)

    model_config = get_cfg(exp_model_config_file)
    logger.info(model_config)
    model_config["epimask"]["match_coarse"]["thr"] = args.match_thr

    # load train_args.yaml, expected as a sibling of model_cfg_path (both are written
    # together by scripts/epimask/train.sh). Released checkpoints under model_weights/
    # don't ship this file, so its absence only skips the img_patch_size check below
    # rather than being a hard error.
    import yaml
    exp_args_file = os.path.join(os.path.dirname(args.model_cfg_path), "train_args.yaml")
    if os.path.exists(exp_args_file):
        train_args = yaml.safe_load(open(exp_args_file, 'r'))
    else:
        logger.info("Warning: train_args.yaml not found at %s -- skipping img_patch_size check" % (exp_args_file))
        train_args = None


    # if args.image_size not same as model_config["img_patch_size"], then override model_config -- scale things
    if train_args is not None and args.img_patch_size != train_args["img_patch_size"]:
        logger.info("Warning: img_patch_size: %d is not same as train arg img_patch_size: %d"%(args.img_patch_size, train_args["img_patch_size"]))
        scale_factor = args.img_patch_size / train_args["img_patch_size"]

        model_config["epimask"]["coarse"]["epi_dist_thr_start"] = int(model_config["epimask"]["coarse"]["epi_dist_thr_start"] * scale_factor)
        model_config["dataset"]["satdepth_img_resize"] = args.img_patch_size
        logger.info("Scaled EPI_DIST_THR_START to %d for image size %d"%(model_config["epimask"]["coarse"]["epi_dist_thr_start"], args.img_patch_size))


    # print args into the out_folder
    args_file = os.path.join(out_folder, "args.txt")
    with open(args_file, "w") as f:
        f.write(str(args))
    f.close()

    # run settings
    epi_thrs = np.linspace(1, 5, 5)
    int_angle_bin_width = 15
    int_angles = np.arange(0, 90, int_angle_bin_width)
    num_int_angle_bins = len(int_angles)
    rel_track_angle_bin_width = 60
    rel_track_angles = np.arange(0, 180, rel_track_angle_bin_width)
    num_rel_track_angle_bins = len(rel_track_angles)

    # run
    (
    p_patches,
    r_patches,
    n_correct_patches,
    n_gt_matches_patches,
    n_detected_patches,
    phi_errs_patches,
    theta_errs_patches,
    s_errs_patches,
    intersection_angles_per_image,
    rel_track_angles_per_image) = run(
                                    args,
                                    model_config,
                                    epi_thrs,
                                    out_folder,
                                    logger)

    summary_filename = os.path.join(out_folder, "summary.pkl")
    summary_data = {
        # 'p': p, #p, # for simulated rot we only do one epoch ie one random patch per image so p and p_patch are the same
        # 'r': r, #r,
        'int_angle_p': None, #int_angle_p,
        'int_angle_r': None, #int_angle_r,
        "int_angles": int_angles,
        "rel_track_angle_p": None, #rel_track_angle_p,
        "rel_track_angle_r": None, #rel_track_angle_r,
        "rel_track_angles": rel_track_angles,
        "p_patches": p_patches,
        "r_patches": r_patches,
        "n_correct_patches": n_correct_patches,
        "n_gt_matches_patches": n_gt_matches_patches,
        "n_detected_patches": n_detected_patches,
        "phi_errs_patches": phi_errs_patches,
        "theta_errs_patches": theta_errs_patches,
        "s_errs_patches": s_errs_patches,
        'intersection_angles_per_image': intersection_angles_per_image,
        'rel_track_angles_per_image': rel_track_angles_per_image
    }

    pkl_write(summary_filename, summary_data)
    