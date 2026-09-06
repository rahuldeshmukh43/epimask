#!/bin/bash
# running script
# add to your bashrc (or rely on the export below) - export PYTHONPATH=<parent-of-epimask-repo>:$PYTHONPATH
# Usage:
# conda activate epimask
# ./test_epimask_simulated_rot.sh <testing-set-name>_simulated_rot <deviceid int (0,1,2,..)> <test pairlist csv> <test shard dir>
# ./test_epimask_simulated_rot.sh testing_set_jacksonville_simulated_rot 0 data/satdepth/index/jax_test_pairs.csv data/satdepth/webdataset/satdepth-hw336-128k-rot_aug/test_whole_image/jacksonville
# *make sure batch_size below and outdir/ckpt_path in test_epimask_simulated_rot.yaml match the run you want to evaluate*

if [ -z "$1" ];
then
    echo "Please provide the testing set foldername"
    exit 1
else
	testing_foldername=$1
fi

if [ -z "$2" ];
then
	device=0
else
	device=$2
fi

if [ -z "$3" ];
then
    echo "Please provide the test pairlist csv file."
    exit 1
else
    test_pairlist=$3
fi

if [ -z "$4" ];
then
    echo "Please provide the test shard dir"
    exit 1
else
    test_shard_dir=$4
fi

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR=$(dirname $(dirname $SCRIPTPATH))

# epimask/ uses absolute imports like `from epimask.src... import ...`, so it's the
# repo's *parent* directory that needs to be on PYTHONPATH, not the repo root itself.
# external/satlaspretrain also needs to be on PYTHONPATH directly, since its own internal
# code uses absolute imports like `from satlaspretrain_models.utils import ...`.
# external/ (not external/satdepth) needs to be on PYTHONPATH too, since satdepth's own
# internal code uses absolute imports like `import satdepth.src.utils...` and the clone is
# itself named "satdepth".
export PYTHONPATH=$(dirname $PROJECT_DIR):$PROJECT_DIR/external/satlaspretrain:$PROJECT_DIR/external:$PYTHONPATH
cd $PROJECT_DIR

yaml_config="$SCRIPTPATH"/"test_epimask_simulated_rot.yaml"

# outdir (the training run to evaluate) is read from test_epimask_simulated_rot.yaml

batch_size=1
num_test_epochs=1
num_workers=4

python -u ./test_epimask_simulated_rot.py \
    --config $yaml_config \
    --test_pairlist $test_pairlist \
    --shard_dir $test_shard_dir \
    --num_workers $num_workers \
    --device $device \
    --testing_foldername $testing_foldername \
    --num_test_epochs $num_test_epochs \
    --batch_size=${batch_size}
