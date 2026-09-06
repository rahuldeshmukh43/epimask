#!/bin/bash
# running script
# conda activate epimask
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
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

TRAIN_IMG_SIZE=336 #[336 for HR(4_2) and 448 for LR (8_2)]
data_cfg_path="configs/data/satdepth_trainval_${TRAIN_IMG_SIZE}.py"

# model design related params go here:
# main_cfg_path="configs/epimask/sat/epimask-LR-gamma-pt4-stage1.py"
# main_cfg_path="configs/epimask/sat/epimask-LR-gamma-pt6-stage1.py"
# main_cfg_path="configs/epimask/sat/epimask-LR-gamma-pt4-lora32-stage2.py"
# main_cfg_path="configs/epimask/sat/epimask-LR-gamma-pt6-lora32-stage2.py"
main_cfg_path="configs/epimask/sat/epimask-HR-gamma-pt4-stage1.py"
# main_cfg_path="configs/epimask/sat/epimask-HR-gamma-pt6-stage1.py"
# main_cfg_path="configs/epimask/sat/epimask-HR-gamma-pt4-lora32-stage2.py"
# main_cfg_path="configs/epimask/sat/epimask-HR-gamma-pt6-lora32-stage2.py"

yaml_config="$SCRIPTPATH"/"train.yaml"

# logdir name should match the main_cfg_path selected above -- see model_weights/ for
# the naming convention (epimask-{HR,LR}-gamma-pt{4,6}[-lora32-stage2])
logdir="${PROJECT_DIR}/training_experiments/epimask-HR-gamma-pt4-stage1"

# * make sure n_gpus_per_node * torch_num_workers = 8
n_nodes=1
n_gpus_per_node=2 # 2 for 2xa5000 4 for 4xa6000
torch_num_workers=4 # 4 for 2xa5000 2 for 4xa6000

# Batch sizes
# 8_2_448: 2 (per a5000)
# 4_2_336: 1 (per gpu a6000) takes >50% of 23-26GB
batch_size=1 # per gpu bs
pin_memory=false 
dataset_type="sharded" # ["sharded" | "default"- on the-fly loading]

log_every_n_steps=96 # make sure that this is multiple of gradient accumulation steps

exp_name="sat-epimask-${TRAIN_IMG_SIZE}-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))"

mkdir -p $logdir

######
#print git branch and log
{
echo 'Git Branch '
git branch -a
echo '=================='
echo 'git log -n 5'
git log -n 5
######

echo '=================='
echo 'Data Config Path'
echo $data_cfg_path
echo 'Main Config Path'
echo $main_cfg_path
echo 'YAML Config Path'
echo $yaml_config
echo '=================='

echo '=================='
echo 'Log Directory'
echo $logdir
echo 'Batch Size'
echo $batch_size
echo 'Number of GPUs per Node'
echo $n_gpus_per_node
echo 'Number of Nodes'
echo $n_nodes
echo 'Number of Torch Workers'
echo $torch_num_workers
echo '=================='


python ./train_epimask.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --config $yaml_config \
    --logdir $logdir \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} \
    --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=${log_every_n_steps} \
    --flush_logs_every_n_steps=100 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --rot_aug \
    --dataset_type $dataset_type \
    --max_epochs=30 2>&1 

echo Finished Training 

} | tee "$logdir"/training.stdout