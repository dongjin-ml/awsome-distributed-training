#!/usr/bin/env bash
set -euxo pipefail

export WORLD_SIZE=$1
export N_PROC=$2
export NODE_RANK=$3
export MASTER_ADDR=$4
export MASTER_PORT=$5

echo "Hello, I am $(hostname), processing rank ${NODE_RANK}."
echo "Set Environment variables for distributed training"
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
export FI_EFA_FORK_SAFE=1
# export NCCL_ALGO=Ring
export FI_LOG_LEVEL=1
export FI_PROVIDER=efa # change to eth if you want to use ENA for comparisons
export FI_EFA_ENABLE_SHM_TRANSFER=1
# https://discuss.pytorch.org/t/nccl-network-is-unreachable-connection-refused-when-initializing-ddp/137352
# https://github.com/pytorch/pytorch/issues/68893
#export NCCL_SOCKET_IFNAME=ens
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

echo "Sanity check"
nvidia-smi
PROFILE_FILE=/fsx/logs/profile

# if [ ${NODE_RANK} -eq 0 ]; then
#     # Use nvidia-nsight only on RANK 0
#     mkdir -p /fsx/logs
#     PROFILER="nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o /fsx/logs/profile-rank-${NODE_RANK}"
# else
#     PROFILER=""
# fi
mkdir -p /fsx/logs
MODEL=mpt-7b
PROFILER="nsys profile --force-overwrite=true  -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o /fsx/logs/profile-mpt30b-rank-${NODE_RANK}"
start=${SECONDS}
nsys profile  --duration 300 --delay 30 --force-overwrite=true -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -x true -o /fsx/logs/${MODEL}-p4d-worldsize-${WORLD_SIZE}-rank-${NODE_RANK}.profile composer \
    --world_size ${WORLD_SIZE} \
    --nproc ${N_PROC} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --verbose /apps/reference-architectures/3.test_cases/3.MPT/llm-foundry/scripts/train/train.py \
    /llm-foundry/scripts/train/yamls/pretrain/${MODEL}.yaml \
    data_local=/fsx/my-copy-c4 \
    train_loader.dataset.split=train_small \
    eval_loader.dataset.split=val_small \
    max_duration=3ba \
    eval_interval=0 \
    save_folder=${MODEL}
end=${SECONDS}
echo "Elapsed Time: $(($end-$start)) seconds"
