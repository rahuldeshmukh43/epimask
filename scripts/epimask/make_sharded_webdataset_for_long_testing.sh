#!/bin/bash
# Builds the whole-image test sharded webdatasets consumed by test_epimask_sharded.py
# (one shard set per AOI test pair, tiled to cover the full image via a DSM grid), via
# make_sharded_webdataset.py --phase test. For train/val shards, see
# make_sharded_webdataset.sh instead -- same python script, different default configs.
#
# Usage:
# conda activate epimask
# ./make_sharded_webdataset_for_long_testing.sh                # builds all configs listed below
# ./make_sharded_webdataset_for_long_testing.sh test_ucsd_336   # builds a single config by name
#
# To add another AOI/size, drop a new yaml under sharded_dataset_configs/
# (see the existing test_*.yaml files for the expected keys) and add its name to CONFIGS below.

set -e

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR=$(dirname $(dirname $SCRIPTPATH))

# epimask/ uses absolute imports like `from epimask.src... import ...`, so it's the
# repo's *parent* directory that needs to be on PYTHONPATH, not the repo root itself.
# external/ needs to be on PYTHONPATH too, since satdepth's own internal code uses absolute
# imports like `import satdepth.src.utils...` and the clone is itself named "satdepth".
export PYTHONPATH=$(dirname $PROJECT_DIR):$PROJECT_DIR/external:$PYTHONPATH
cd $PROJECT_DIR

CONFIG_DIR="$PROJECT_DIR/src/datasets/sharded_dataset_configs"

CONFIGS=(
    test_jacksonville_336
    test_omaha_336
    test_ucsd_336
    test_argentina_336
    test_jacksonville_448
    test_omaha_448
    test_ucsd_448
    test_argentina_448
)

if [ -n "$1" ]; then
    CONFIGS=("$1")
fi

for cfg_name in "${CONFIGS[@]}"; do
    cfg_path="$CONFIG_DIR/${cfg_name}.yaml"
    echo '=================='
    echo "Building shards for config: $cfg_path"
    echo '=================='
    python -u src/datasets/make_sharded_webdataset.py --config "$cfg_path"
done
