#!/bin/bash
# Usage : bash scripts/general/train.sh <corpus> <model_size> <subtask>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <corpus> <model_size> <subtask>"
    echo "Example: $0 rstdt 7b span"
    exit 1
else
    CORPUS=$1
    MODEL_NAME=$2
    SUBTASK=$3

    echo "===================="
    echo "Corpus: ${CORPUS}"
    echo "Model: ${MODEL_NAME}"
    echo "Subtask: ${SUBTASK}"
    echo "===================="
fi

sleep 5

export LD_LIBRARY_PATH=/usr/local/cuda/lib64

python src/train.py \
    --model_name meta-llama/Llama-2-${MODEL_NAME}-hf \
    --output_dir ./outputs/${CORPUS}-${SUBTASK}-${MODEL_NAME} \
    --dataset_dir preprocessed_data/${CORPUS}/train \
    --train_file ${SUBTASK}.json \
    --dataset_format input-output \
    --data_seed 42 \
    --use_auth \
    --num_train_epochs 5 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --save_total_limit 5 \
    --dataloader_num_workers 4 \
    --group_by_length \
    --logging_strategy steps \
    --logging_steps 100 \
    --remove_unused_columns False \
    --do_train \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --source_max_len 4000 \
    --target_max_len 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --load_in_4bit \
    --use_peft \
    --report_to wandb
