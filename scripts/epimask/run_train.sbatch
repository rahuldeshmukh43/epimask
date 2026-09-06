#!/bin/bash
#
#SBATCH --job-name=satdepth_4_2_336          # job name
#SBATCH --output=/dev/null
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4                # ask Slurm for 4 GPUs
#SBATCH --mem=96G
#SBATCH --time=48:00:00


# activate your conda env -- update this to point at your own anaconda3/miniconda3 install
__conda_setup="$('/mnt/cloudNAS3/rahul/Installations/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

eval "$__conda_setup"
conda activate epimask

echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

echo "Launching training script..."

srun ./train.sh

echo "Exited at: $(date)"