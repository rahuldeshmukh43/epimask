#!/bin/bash
# running script
# add to your bashrc (or rely on the export below) - export PYTHONPATH=<parent-of-epimask-repo>:$PYTHONPATH
# Usage:
# conda activate epimask
# ./test_epimask.sh <testing-set-name> <deviceid int (0,1,2,..)>
# ./test_epimask.sh testing_set_jacksonville 0

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

yaml_config="$SCRIPTPATH"/"test_epimask.yaml"

# outdir (the training run to evaluate) is read from test_epimask.yaml

batch_size=6

python -u ./test_epimask.py \
    --config $yaml_config \
    --device $device \
    --testing_foldername $testing_foldername \
    --do_match_plot \
    --batch_size=${batch_size}
