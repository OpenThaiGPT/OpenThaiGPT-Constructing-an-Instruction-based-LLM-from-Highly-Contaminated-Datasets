

# source /opt/cray/pje/cpe/23.09/restore_lmod_system_defaults.sh
module purge
module load Miniconda3/22.11.1-1
# module load cpe-cuda/23.03
module load cudatoolkit/23.3_11.8
module load gcc/11.2.0
module load PrgEnv-nvidia
# module load gcc/11.2
# module load PrgEnv-gnu
# module load cpe-cuda
# module load cudatoolkit/22.7_11.7
# module load craype-accel-nvidia80
# module load aws-ofi-nccl

TRAIN_DATA_DIR=/workspace/train
VAL_DATA_DIR=/workspace/val

export WANDB_MODE=offline
srun python pretrain/tinyllama.py \
    --train_data_dir $TRAIN_DATA_DIR \
    --val_data_dir $VAL_DATA_DIR \
    --devices 4 \
    --num_nodes 10 \

