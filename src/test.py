import json
import os
from datetime import datetime

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from tap import Tap
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from data.tree import AttachTree
from metrics import OriginalParseval, Parseval, RSTParseval
from parse.parse import parse_dataset


class Config(Tap):
    # pretrained model
    base_model_name: str = "meta-llama/Llama-2-7b-hf"

    span_lora_params: str | None = None
    nuc_lora_params: str | None = None
    rel_lora_params: str | None = None
    nuc_rel_lora_params: str | None = None
    rel_with_nuc_lora_params: str | None = None

    top_down_lora_params: str | None = None

    zero_shot: bool = False

    # data
    data_dir: str = "data"
    corpus: str = "rstdt"  # rstdt or instrdt
    valid_file: str = "valid.json"
    test_file: str = "test.json"

    # test
    save_result_dir: str = "results/rstdt-7b"
    save_dir_name: str | None = None

    parse_type: str = "bottom_up"  # bottom_up or top_down
    rel_type: str = "rel"  # rel or nuc_rel or rel_with_nuc

    skip_valid: bool = False

    # debug
    max_examples: int = -1


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


# load base model
def load_model(
    config: Config, model_type_list: list[str]
) -> tuple[PeftModel, AutoTokenizer]:
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # resize tokenizer and embedding
    smart_tokenizer_and_embedding_resize({"pad_token": "[PAD]"}, tokenizer, model)

    # peft model with lora adapters
    peft_model = None
    for model_type in model_type_list:
        if config.zero_shot:
            dummy_peft_config = LoraConfig(
                r=0,
                target_modules=None,
                task_type="CAUSAL_LM",
                bias="none",
            )
            if peft_model is None:
                peft_model = get_peft_model(
                    model, dummy_peft_config, adapter_name=model_type
                )
            else:
                peft_model.add_adapter(model_type, dummy_peft_config)

        else:
            params_path = getattr(config, f"{model_type}_lora_params")
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(
                    model,
                    params_path,
                    adapter_name=model_type,
                    # device_map={"": 0},
                )
            else:
                peft_model.load_adapter(params_path, model_type)

    peft_model.eval()

    return peft_model, tokenizer


def compute_metrics(
    metrics: Parseval | tuple[Parseval], output: dict
) -> dict[str, float]:
    if isinstance(metrics, Parseval):
        metrics = (metrics,)

    all_score = {}
    for metric in metrics:
        metric.update(output["pred_tree"], output["gold_tree"])
        score = metric.compute()
        assert all(key not in all_score.keys() for key in score.keys())
        all_score |= {k: v.item() for k, v in score.items()}
        metric.reset()

    assert all(isinstance(v, float) for v in all_score.values())

    return all_score


def save_tree(output: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "pred"), exist_ok=True)
    for doc_id, tree in zip(output["doc_id"], output["pred_tree"]):
        assert isinstance(tree, AttachTree)
        with open(os.path.join(save_dir, "pred", f"{doc_id}.tree"), "w") as f:
            print(tree, file=f)

    os.makedirs(os.path.join(save_dir, "gold"), exist_ok=True)
    for doc_id, tree in zip(output["doc_id"], output["gold_tree"]):
        assert isinstance(tree, AttachTree)
        with open(os.path.join(save_dir, "gold", f"{doc_id}.tree"), "w") as f:
            print(tree, file=f)
    return


def get_model_type_list(config: Config):
    if config.parse_type == "bottom_up":
        model_type_list = ["span"]
    elif config.parse_type == "top_down":
        model_type_list = ["top_down"]
    else:
        raise ValueError(f"Unknown parse_type: {config.parse_type}")

    if config.rel_type == "rel":
        model_type_list += ["nuc", "rel"]
    elif config.rel_type == "rel_with_nuc":
        model_type_list += ["nuc", "rel_with_nuc"]
    elif config.rel_type == "nuc_rel":
        model_type_list += ["nuc_rel"]
    else:
        raise ValueError(f"Unknown rel_type: {config.rel_type}")

    # check existence of lora params
    for model_type in model_type_list:
        if not config.zero_shot:
            assert (
                getattr(config, f"{model_type}_lora_params") is not None
            ), f"{model_type}_lora_params"
            # assert os.path.exists(getattr(config, f"{model_type}_lora_params"))

    return model_type_list


def test(config: Config):
    debug = config.max_examples != -1

    # save
    timestamp = datetime.now().strftime("%Y-%m-%d.%H-%M-%S")

    if config.save_dir_name is None:
        config.save_dir_name = timestamp
    save_result_dir = os.path.join(config.save_result_dir, config.save_dir_name)

    print("=" * 80)
    print(f"save_result_dir: {save_result_dir}")
    print("=" * 80)
    os.makedirs(save_result_dir, exist_ok=False)
    with open(os.path.join(save_result_dir, "timestamp.txt"), "w") as f:
        f.write(timestamp)

    config.save(os.path.join(save_result_dir, "config.json"))
    print("=" * 80)
    print("Config: ", json.dumps(config.as_dict(), indent=4))
    print("=" * 80)

    # check
    model_type_list = get_model_type_list(config)

    print("=" * 80)
    print(f"model_type_list: {json.dumps(model_type_list, indent=4)}")

    # load model
    model, tokenizer = load_model(config, model_type_list)

    # metric
    metric = (RSTParseval(), OriginalParseval())

    # valid data
    if not config.skip_valid:
        print(" Valid data ".center(80, "="))
        valid_data = json.load(
            open(os.path.join(config.data_dir, config.corpus, config.valid_file))
        )
        if debug:
            valid_data = valid_data[5 : 5 + config.max_examples]

        output = parse_dataset(
            valid_data,
            model,
            tokenizer,
            parse_type=config.parse_type,
            rel_type=config.rel_type,
            corpus=config.corpus,
        )
        valid_scores = compute_metrics(metric, output)
        print("valid scores:", json.dumps(valid_scores, indent=4))
        valid_scores_file = os.path.join(save_result_dir, "valid_scores.json")
        # save result
        json.dump(valid_scores, open(valid_scores_file, "w"), indent=4)
        save_tree(output, os.path.join(save_result_dir, "valid_trees"))

    if debug:
        return

    # test data
    print(" Test data ".center(80, "="))
    test_data = json.load(
        open(os.path.join(config.data_dir, config.corpus, config.test_file))
    )

    output = parse_dataset(
        test_data,
        model,
        tokenizer,
        parse_type=config.parse_type,
        rel_type=config.rel_type,
        corpus=config.corpus,
    )
    test_scores = compute_metrics(metric, output)
    print("test scores:", json.dumps(test_scores, indent=4))
    test_scores_file = os.path.join(save_result_dir, "test_scores.json")
    # save result
    json.dump(test_scores, open(test_scores_file, "w"), indent=4)
    save_tree(output, os.path.join(save_result_dir, "test_trees"))


if __name__ == "__main__":
    config = Config().parse_args()
    test(config)
