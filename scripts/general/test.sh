#!/bin/bash
# Usage : bash scripts/general/test.sh <corpus (train)> <corpus (test)> <model_name> <parse_type>

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <corpus (train)> <corpus (test)> <model_name> <parse_type>"
    echo "Example: $0 rstdt rstdt 7b bottom_up"
    exit 1
else
    corpus_train=$1
    corpus_test=$2
    model_name=$3
    parse_type=$4
    echo "===================="
    echo "Corpus (train): ${corpus_train}" # rstdt, instrdt, gum
    echo "Corpus (test): ${corpus_test}"   # rstdt, instrdt, gum
    echo "Model: ${model_name}"            # 7b, 13b, 70b
    echo "Parse type: ${parse_type}"       # bottom_up, top_down
    echo "===================="
fi

sleep 5

python src/test.py \
    --base_model_name meta-llama/Llama-2-${model_name}-hf \
    --span_lora_params outputs/${corpus_train}-span-${model_name}/checkpoint-best/adapter_model \
    --top_down_lora_params outputs/${corpus_train}-top_down-${model_name}/checkpoint-best/adapter_model \
    --nuc_lora_params outputs/${corpus_train}-nuc-${model_name}/checkpoint-best/adapter_model \
    --rel_with_nuc_lora_params outputs/${corpus_train}-rel-with-nuc-${model_name}/checkpoint-best/adapter_model \
    --data_dir data \
    --corpus ${corpus_test} \
    --save_result_dir results/${corpus_test} \
    --parse_type ${parse_type} \
    --rel_type "rel_with_nuc" \
    --save_dir_name Llama-2-${model_name}-${corpus_train}-${parse_type}
