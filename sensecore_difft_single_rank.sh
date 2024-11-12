#!/bin/bash
EXP=${1:-cogvx-difft}
MODEL_PATH=${2:-models/CogVideoX-2b}
DATASET_PATH=${3:-datas/Disney}
OUTPUT_PATH=${4:-lora-single-node}

HOME_DIR=$(pwd)
CACHE_PATH="models/.cache"
DATETIME=$(date '+%Y-%m-%d-%H:%M:%S')
environs="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo ${HOME_DIR}
echo ${EXP}
echo ${MODEL_PATH}
echo ${DATASET_PATH}
echo ${OUTPUT_PATH}
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
    bash -c "cd \"${HOME_DIR}\"; \
    source /root/miniconda3/bin/activate cogvx_diff; \
    $environs accelerate launch --config_file finetune/accelerate_config_machine_single.yaml --multi_gpu \
    finetune/train_cogvideox_lora.py \
    --gradient_checkpointing \
    --pretrained_model_name_or_path $MODEL_PATH \
    --cache_dir $CACHE_PATH \
    --enable_tiling \
    --enable_slicing \
    --instance_data_root $DATASET_PATH \
    --caption_column prompts.txt \
    --video_column videos.txt \
    --validation_prompt \"A young woman with pink hair and a sailor-style uniform stands confidently, gesturing animatedly. In a mystical forest, a unicorn with a spiraled horn stands amidst lush greenery and a clear stream. A young girl with pink hair and a whimsical hat poses dynamically against a soft pink backdrop:::DISNEY A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions:::A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance\" \
    --validation_prompt_separator ::: \
    --num_validation_videos 1 \
    --validation_epochs 5 \
    --seed 42 \
    --rank 128 \
    --lora_alpha 64 \
    --mixed_precision fp16 \
    --output_dir save_${OUTPUT_PATH}_${DATETIME} \
    --height 480 \
    --width 720 \
    --fps 8 \
    --max_num_frames 49 \
    --skip_frames_start 0 \
    --skip_frames_end 0 \
    --train_batch_size 2 \
    --num_train_epochs 30 \
    --checkpointing_epochs 5 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-3 \
    --lr_scheduler cosine_with_restarts \
    --lr_warmup_steps 200 \
    --lr_num_cycles 1 \
    --enable_slicing \
    --enable_tiling \
    --gradient_checkpointing \
    --optimizer AdamW \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --allow_tf32 \
    --use_dynamic_cfg \
    --report_to tensorboard;\
  sleep 1d"

echo "DONE on `hostname`"
# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number