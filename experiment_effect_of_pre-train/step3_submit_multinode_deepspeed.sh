


export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn 
export NCCL_P2P_DISABLE=1
#export FI_MR_CACHE_MONITOR=memhooks
#export NCCL_NET_GDR_LEVEL=3
#export NCCL_NET=IB
#export NCCL_IB_HCA=mlx5
#export CXI_FORK_SAFE=1 
#export CXI_FORK_SAFE_HP=1 
#export FI_CXI_DISABLE_CQ_HUGETLB=1

#echo "using FI_MR_CACHE_MONITOR=memhooks"

START=`date`
starttime=$(date +%s)

export WANDB_MODE="offline"

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

srun sh step3_smultinode_deepspeed.sh

current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;