#!/bin/bash
#$ -M darun@nd.edu
#$ -m ae
#$ -q gpu
#$ -l h=qa-a10-*            # A10 nodes only
#$ -l gpu_card=2            # <-- set how many GPUs you want (2, 3, or 4)
#$ -pe smp 24
#$ -N fm
#$ -o /afs/crc.nd.edu/user/d/darun/if-copy2/recognition/arcface_torch/logs_l

cd /users/darun/afs/conditional-flow-matching/

module load conda
source activate ft

# -----------------------------------------------------------------------------------
# Multi-GPU environment setup (A10 compute capability = 8.6)
# -----------------------------------------------------------------------------------
export TORCH_CUDA_ARCH_LIST="8.6"
export OMP_NUM_THREADS=${NSLOTS:-8}
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export PYTHONUNBUFFERED=1

# Number of GPUs the scheduler assigned:
NPROC=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | awk -F, '{print NF}')

echo "Node hostname: $(hostname)"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Launching training with $NPROC GPUs"
echo "Start: $(date)"

start_time=$(date +%s)

# -----------------------------------------------------------------------------------
# Launch Distributed Data Parallel
# -----------------------------------------------------------------------------------
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" train_ddp.py

status=$?
echo "Finished: $(date) (exit code $status)"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf "Total time: %02d:%02d:%02d\n" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
exit $status

