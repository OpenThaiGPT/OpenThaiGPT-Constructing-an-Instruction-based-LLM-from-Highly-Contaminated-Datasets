#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDM

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script



module restore
module load Miniconda3
module load PrgEnv-gnu
module load cpe-cuda
module load cudatoolkit/22.7_11.7
module load craype-accel-nvidia80
# module load aws-ofi-nccl
module load gcc/10.3.0


conda deactivate
conda activate /project/lt200056-opgpth/boss/stanford_alpaca/env

# conda deactivate
# conda activate /project/lt200056-opgpth/multinode-fix/stanford_alpaca_init/conda

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_ADDR
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
THEID=`echo -e $HOSTNAMES | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID
echo SLURM_PROCID=$SLURM_PROCID

export NCCL_TIMEOUT=3600000
export NCCL_BLOCKING_WAIT=0


# source /fsx/dalle2/.dalle_env_38/bin/activate
# echo python3 version = `python3 --version`
# python -c "import torch"

MODEL_DIR=/workspace/model
TOKENIZER_DIR=/workspace/data
OUTPUT_DIR=/workspace/output
DATA_THA_DIR=/workspace/tha
DATA_ENG_DIR=/workspace/en


accelerate launch \
    --num_processes $(( 4 * $COUNT_NODE )) \
    --num_machines $COUNT_NODE \
    --multi_gpu \
    --mixed_precision fp16 \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    ./train_v2.py \
        --model_name_or_path $MODEL_DIR \
        --tokenizer_name_or_path $TOKENIZER_DIR \
        --use_flash_attention_2 False \
        --data_path $DATA_THA_DIR \
        $DATA_ENG_DIR \
        --data_weights 0.9 0.1 \
        --data_seed 42 \
        --train_split train \
        --eval_split eval \
        --bf16 True \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "steps" \
        --eval_steps 700 \
        --save_strategy "steps" \
        --save_steps 700 \
        --save_total_limit 5 \
        --logging_strategy 'steps' \
        --logging_steps 1 \
        --logging_first_step True \
        --learning_rate 5e-5 \
        --weight_decay 0.001 \
        --warmup_ratio 0.03 \
        --deepspeed ../src/model/scripts/hf_trainer/config/llama_deepspeed.json \
        --tf32 True \
        --gradient_checkpointing True \
        --max_grad_norm 1.00 \
        --lr_scheduler_type cosine

    
    
    # --checkpoint /project/lt200056-opgpth/weight_llama_2_finetune_7b_512_th100/checkpoint-250 \
    
    # --use_flash_attention_2 True \

    # --fsdp "full_shard auto_wrap" \
    # --gradient_checkpointing True