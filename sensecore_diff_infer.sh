#! /bin/bash
HOME_DIR=$(pwd)
EXP=${1:-cogvx-diff-infer}
# PROMPT=${2:-A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship\'s hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children\'s items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship\'s journey symbolizing endless adventures in a whimsical, indoor setting.}
PROMPT=${2:-A young woman with pink hair and a sailor-style uniform stands confidently, gesturing animatedly. In a mystical forest, a unicorn with a spiraled horn stands amidst lush greenery and a clear stream. A young girl with pink hair and a whimsical hat poses dynamically against a soft pink backdrop.}
MODEL_PATH=${3:-models/CogVideoX-5b}
GEN_TYPE=${4:-t2v}
DATETIME=$(date '+%Y-%m-%d-%H:%M:%S')

echo ${HOME_DIR}
echo ${EXP}
echo ${PROMPT}
echo ${MODEL_PATH}
echo ${GEN_TYPE}
echo ${DATETIME}

srun --partition-id share-a \
    --workspace-id d08f360b-7f9c-4eb1-bfb3-155fdad18726 \
    --framework pt \
    --job-name ${EXP} \
    --resource N3lS.Ii.I60.1 \
    --distributed StandAlone \
    --output run_${DATETIME}.log \
    --nodes 1 \
    --priority highest \
    --container-image registry.cn-sh-01.sensecore.cn/devsft-ccr/ubuntu22.04_cuda12.4_cogvx:v2.1.5 \
    --container-mounts 4ba8dc8e-52e5-11ee-82fd-de3a99f44f33:/mnt/afs_1 \
    bash -c "cd \"${HOME_DIR}\"; source /root/miniconda3/bin/activate cogvx_diff; python inference/cli_demo.py \
    --prompt \"${PROMPT}\" \
    --model_path ${MODEL_PATH} \
    --generate_type \"${GEN_TYPE}\" \
    --dtype bfloat16; \
    sleep 1d"

echo "DONE on `hostname`"