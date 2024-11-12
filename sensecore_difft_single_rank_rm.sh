#!/bin/bash
EXP=${1:-cogvx-difft-rm}
MODEL_PATH=${2:-models/CogVideoX-2b}
DATASET_PATH=${3:-datas/Dance}
OUTPUT_PATH=${4:-reward-single-node}

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
    $environs accelerate launch --config_file finetune/accelerate_config_machine_single_zero3_1.yaml --multi_gpu\
    finetune/train_cogvideox_lora_reward_cmb.py \
    --gradient_checkpointing \
    --pretrained_model_name_or_path $MODEL_PATH \
    --cache_dir $CACHE_PATH \
    --enable_tiling \
    --enable_slicing \
    --instance_data_root $DATASET_PATH \
    --caption_column prompts.txt \
    --video_column videos.txt \
    --validation_prompt \"A young woman with pink hair and a sailor-style uniform stands confidently, gesturing animatedly. In a mystical forest, a unicorn with a spiraled horn stands amidst lush greenery and a clear stream. A young girl with pink hair and a whimsical hat poses dynamically against a soft pink backdrop.:::Tom, the mischievous gray cat, is crouched in a barn, his eyes gleaming with cunning as he prepares to pounce on Jerry, the agile white mouse. The barn is filled with rustic charm, featuring wooden beams and a hay bale in the background. Tom's body language is tense, his muscles coiled like a spring, ready to leap at any moment. Jerry, on the other hand, is caught off guard, standing on his hind legs with a look of surprise on his face. The scene is filled with anticipation, as viewers wait to see if Tom will finally catch Jerry. The colors are vibrant, with the gray of Tom's fur contrasting against the white of Jerry's fur and the warm tones of the barn. The composition of the scene is dynamic, with Tom and Jerry positioned diagonally across from each other, creating a sense of movement and tension.:::A young woman with long dark hair adorned with a white bow stands against a beige backdrop. She wears an elegant off-the-shoulder dress with intricate lace and a high ruffled hem. Her posture is poised, inviting the viewer. The dress is crafted from layers of sheer tulle, creating a dreamy effect. Her serene pose and gentle expression add to the ethereal quality of the scene.
:::DISNEY A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions:::A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance:::a tortoise covered with algae\" \
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
    --train_batch_size 1 \
    --num_train_epochs 100 \
    --checkpointing_epochs 5 \
    --gradient_accumulation_steps 4 \
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