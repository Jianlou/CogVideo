#! /bin/bash
HOME_DIR=${0%/*}
cd ${HOME_DIR}
HOME_DIR=$(pwd)
EXP=${1:-cogvx-train-single}
MODEL_CONFIG=${2:-cogvideox_rm_2b_lora}
RUN_CONFIG=${3:-sft_2b}
DATETIME=$(date '+%Y-%m-%d-%H:%M:%S')
environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ${HOME_DIR}
echo ${EXP}
echo ${DATETIME}
echo ${environs}

srun --partition-id share-a \
    --workspace-id d08f360b-7f9c-4eb1-bfb3-155fdad18726 \
    --framework pt \
    --job-name ${EXP} \
    --resource N3lS.Ii.I60.1 \
    --distributed StandAlone \
    --output run_${DATETIME}.log \
    --nodes 1 \
    --priority highest \
    --container-image registry.cn-sh-01.sensecore.cn/devsft-ccr/ubuntu22.04_cuda12.4_cogvx:v1.0.6 \
    --container-mounts 4ba8dc8e-52e5-11ee-82fd-de3a99f44f33:/mnt/afs_1 \
    bash -c "cd "${HOME_DIR}"; source /root/miniconda3/bin/activate cogvx_sat; $environs python train_video.py --base configs/${MODEL_CONFIG}.yaml configs/${RUN_CONFIG}.yaml --seed $RANDOM; sleep 1d"

echo "DONE on `hostname`"