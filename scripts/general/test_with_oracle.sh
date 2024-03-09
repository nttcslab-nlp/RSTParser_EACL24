#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <corpus> <model_name> <subtask>"
    echo "Example: $0 rstdt 7b span"
    exit 1
else
    corpus=$1
    model_name=$2
    subtask=$3
    echo "===================="
    echo "Corpus: ${corpus}"
    echo "Model: ${model_name}"
    echo "Subtask: ${subtask}"
    echo "===================="
fi

sleep 5

export LD_LIBRARY_PATH=/usr/local/cuda/lib64

python src/test_with_oracle.py \
    --base_model_name meta-llama/Llama-2-${model_name}-hf \
    --${subtask//-/_}_lora_params outputs/${corpus}-${subtask}-${model_name} \
    --data_dir preprocessed_data \
    --corpus ${corpus} \
    --save_result_dir results_with_oracle/${corpus}-${model_name}

# --nuc_rel_lora_params outputs/rstdt-nuc-rel-${model_name} \
