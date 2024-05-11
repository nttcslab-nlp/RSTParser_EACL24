import json
import os
import shutil

import torch
from datasets import Dataset
from peft import PeftModel
from tap import Tap
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

SUB_TASKS = [
    "span",
    "nuc",
    "rel",
    "rel-with-nuc",
    "top_down",
]


class Config(Tap):
    # pretrained model
    base_model_name: str
    span_lora_params_dir: str | None = None
    nuc_lora_params_dir: str | None = None
    rel_lora_params_dir: str | None = None
    nuc_rel_lora_params_dir: str | None = None
    rel_with_nuc_lora_params_dir: str | None = None

    top_down_lora_params_dir: str | None = None

    # data
    data_dir: str = "preprocessed_data"
    corpus: str = "rstdt"  # rstdt or instrdt

    # save
    save_result_dir: str
    skip_if_exist: bool = True
    test_all_checkpoints: bool = False  # `False` means only test the best checkpoint.

    # others
    batch_size: int = 1


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
def load_model(config: Config) -> tuple[PeftModel, AutoTokenizer]:
    """load model
    Returns:
        model: (PeftModel)
        tokenizer: (PreTrainedTokenizer)
        adapter_path_map: (dict[str, dict[str, str]])
            adapter_path_map[subtask][adapter_name] = "path/to/adapter_params"

    """

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
    adapter_path_map = {}

    for subtask_type in SUB_TASKS:
        subtask_type = subtask_type.replace("-", "_")
        dirname = getattr(config, f"{subtask_type}_lora_params_dir")
        if dirname is None:
            print(f"Skip {subtask_type}")
            continue

        adapter_path_map[subtask_type] = {}

        ckpt_list = [
            ckpt_name
            for ckpt_name in os.listdir(dirname)
            if os.path.isdir(os.path.join(dirname, ckpt_name))
        ]
        ckpt_list = [
            ckpt_name
            for ckpt_name in ckpt_list
            if ckpt_name != "checkpoint-best" and ckpt_name != "checkpoint-avg"
        ]
        ckpt_list.sort(key=lambda ckpt_name: int(ckpt_name.split("checkpoint-")[-1]))
        for ckpt_name in ckpt_list:
            adapter_name = f"{subtask_type}_{ckpt_name}"
            adapter_params_path = os.path.join(dirname, ckpt_name, "adapter_model")
            adapter_path_map[subtask_type][adapter_name] = adapter_params_path

    print(json.dumps(adapter_path_map, indent=4))

    model.eval()

    return model, tokenizer, adapter_path_map


def load_data(
    data_dir: str, corpus: str, split: str = "valid", subtasks: list[str] | None = None
) -> dict[str, Dataset]:
    """load data

    Args:
        data_dir (str): data directory
        corpus (str): rstdt or instrdt
        file_name (str): valid.json or test.json

    Returns:
        _type_:
    """
    if subtasks is None:
        subtasks = SUB_TASKS
    dataset_path_dict = {
        sub_task.replace("-", "_"): os.path.join(
            data_dir, corpus, split, f"{sub_task}.json"
        )
        for sub_task in subtasks
    }

    for path in dataset_path_dict.values():
        assert os.path.exists(path), f"{path} does not exist."

    return {
        key: Dataset.from_json(model_path)
        for key, model_path in dataset_path_dict.items()
    }


def predict(
    data: Dataset, model: PeftModel, tokenizer: AutoTokenizer, batch_size: int = 1
) -> tuple[list, list]:
    """predict

    Args:
        data (list[dict]): data
        model (PeftModel): model
        tokenizer (AutoTokenizer): tokenizer

    Returns:
        list[str]: pred_outputs
        list[str]: gold_outputs
    """
    # # sort by input length
    # data = data.map(lambda ex: {"input_length": len(ex["input"])})
    # data = data.sort("input_length", reverse=True)

    model.eval()
    pred_outputs = []
    for i in tqdm(range(0, len(data), batch_size), dynamic_ncols=True, leave=False):
        batch = data[i : i + batch_size]
        batch_inputs = batch["input"]
        batch_outputs = batch["output"]

        inputs = tokenizer(
            batch_inputs,
            padding=True,
            return_tensors="pt",
            max_length="longest",
        )
        with torch.inference_mode():
            outputs = model.generate(
                **{k: v.cuda() for k, v in inputs.items()}, max_new_tokens=10
            )
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        assert all(
            out_sample.startswith(input_sample)
            for out_sample, input_sample in zip(output_text, batch_inputs)
        )

        answers = [
            out_text[len(in_text) :].strip()
            for in_text, out_text in zip(batch_inputs, output_text)
        ]

        assert len(answers) == len(batch_outputs)
        pred_outputs += answers

    gold_outputs = data["output"]
    assert len(pred_outputs) == len(gold_outputs)

    return pred_outputs, gold_outputs


def test(
    data: dict[str, Dataset],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    adapter_path_map: dict[str, dict[str, str]],
    config: Config,
    result_dir: str,
) -> dict[str, dict]:
    """test

    Args:
        data (dict[str, Dataset]): dataset for each subtask
        model (PeftModel): peft model
        tokenizer (PreTrainedTokenizer): tokenizer
        adapter_path_map (dict[str, list[str]]): peft adapter map

    Returns:
        dict[str, dict]: results dictionary: {
            adapter_name: {"accuracy": (float), "match": (int), "total": (int)},
            ...
        }

    Examples:
        >>> data = load_data("data", "rstdt", "test.json")
        >>> model, tokenizer, adapter_path_map = load_model("peft_model")
        >>> test(data, model, tokenizer, adapter_path_map)
    """

    results_file_path = os.path.join(result_dir, "results.json")
    if config.skip_if_exist and os.path.exists(results_file_path):
        print(f"Load pre-computed results from {results_file_path}")
        results = json.load(open(results_file_path))
    else:
        os.makedirs(result_dir, exist_ok=True)
        results = {}

    def test_subtask(subtask_name: str, adapter_name: str) -> tuple[dict, list[dict]]:
        """test subtask with each adapter"""
        assert os.path.exists(
            adapter_path_map[subtask_name][adapter_name]
        ), adapter_path_map[subtask_name][adapter_name]

        peft_model: PeftModel = PeftModel.from_pretrained(
            model,
            adapter_path_map[subtask_name][adapter_name],
            adapter_name=adapter_name,
            # device_map={"": 0},
        )
        peft_model.set_adapter(adapter_name)

        pred_outputs, gold_outputs = predict(
            data[subtask_name], peft_model, tokenizer, batch_size=config.batch_size
        )

        del peft_model
        torch.cuda.empty_cache()

        num_total = len(pred_outputs)
        num_match = sum(pred == gold for pred, gold in zip(pred_outputs, gold_outputs))

        results = {
            "accuracy": num_match / num_total,
            "match": num_match,
            "total": num_total,
        }

        pred_outputs = [
            {"input": example["input"], "output": pred_text}
            for example, pred_text in zip(data[subtask_name], pred_outputs)
        ]

        return results, pred_outputs

    print(json.dumps(adapter_path_map, indent=4))

    best_models = {}
    for subtask_name in adapter_path_map.keys():
        # save gold data
        save_outputs_path = os.path.join(
            result_dir, subtask_name, f"gold_{subtask_name}.json"
        )
        data[subtask_name].to_json(save_outputs_path)

        print(f"SUBTASK: {subtask_name}")
        print(f"MODEL: {json.dumps(adapter_path_map[subtask_name], indent=4)}")
        best_accuracy = 0
        for adapter_name in adapter_path_map[subtask_name].keys():
            if adapter_name not in results:
                results[adapter_name], pred_outputs = test_subtask(
                    subtask_name, adapter_name
                )
                save_outputs_path = os.path.join(
                    result_dir, subtask_name, f"pred_{adapter_name}.json"
                )
                Dataset.from_list(pred_outputs).to_json(save_outputs_path)

            print(f"MODEL: {adapter_name}, Score: {results[adapter_name]}")

            if best_accuracy < results[adapter_name]["accuracy"]:
                best_models[subtask_name] = adapter_name
                best_accuracy = results[adapter_name]["accuracy"]

            add_to_json_file(results, results_file_path)

        if len(adapter_path_map[subtask_name]) < 5:
            continue

        # copy best ckpt as `checkpoint-best`
        best_ckpt_dir = os.path.dirname(
            adapter_path_map[subtask_name][best_models[subtask_name]]
        )
        save_best_ckpt_path = os.path.join(
            os.path.dirname(best_ckpt_dir), "checkpoint-best"
        )
        if f"{subtask_name}_checkpoint-best" not in results:
            if not os.path.exists(save_best_ckpt_path):
                shutil.copytree(
                    best_ckpt_dir,
                    save_best_ckpt_path,
                )
            results[f"{subtask_name}_checkpoint-best"] = results[
                best_models[subtask_name]
            ]

        adapter_path_map[subtask_name][f"{subtask_name}_checkpoint-best"] = (
            os.path.join(save_best_ckpt_path, "adapter_model")
        )

        print(
            f"MODEL: {subtask_name}_checkpoint-best, "
            f"Score: {results[f'{subtask_name}_checkpoint-best']}"
        )
        add_to_json_file(results, results_file_path)

        # checkpoint averaging
        avg_adapter_name = f"{subtask_name}_checkpoint-avg"
        avg_ckpt_path = os.path.join(os.path.dirname(best_ckpt_dir), "checkpoint-avg")
        adapter_path_map[subtask_name][avg_adapter_name] = os.path.join(
            avg_ckpt_path, "adapter_model"
        )
        if avg_adapter_name not in results:
            if not os.path.exists(avg_ckpt_path):
                # create avg ckpt
                # avg_ckpt_list = list(adapter_path_map[subtask_name].keys())[-3:]
                avg_ckpt_list = []
                for ckpt_name in list(adapter_path_map[subtask_name].keys())[::-1]:
                    if ckpt_name[len(f"{subtask_name}_checkpoint-") :] in {
                        "best",
                        "avg",
                    }:
                        continue

                    avg_ckpt_list.append(ckpt_name)

                    if len(avg_ckpt_list) == 3:
                        avg_ckpt_list = avg_ckpt_list[::-1]
                        break
                assert len(avg_ckpt_list) == 3, avg_ckpt_list

                avg_params_dict = average_ckpt(
                    avg_ckpt_list, adapter_path_map[subtask_name]
                )
                # save avg ckpt
                os.makedirs(avg_ckpt_path, exist_ok=True)
                shutil.copytree(
                    os.path.join(best_ckpt_dir, "adapter_model"),
                    os.path.join(avg_ckpt_path, "adapter_model"),
                )
                torch.save(
                    avg_params_dict,
                    os.path.join(avg_ckpt_path, "adapter_model", "adapter_model.bin"),
                )

            results[avg_adapter_name], pred_outputs = test_subtask(
                subtask_name, avg_adapter_name
            )
            save_outputs_path = os.path.join(
                result_dir, subtask_name, f"pred_{avg_adapter_name}.json"
            )
            Dataset.from_list(pred_outputs).to_json(save_outputs_path)

        print(f"MODEL: {avg_adapter_name}, Score: {results[avg_adapter_name]}")

        add_to_json_file(results, results_file_path)

    return results


def average_ckpt(ckpt_list: list[str], adapter_path_map: dict[str, list[str]]):
    print(f"Average checkpoints: {ckpt_list}")
    adapter_param_keys = None
    avg_params_dict = {}
    for ckpt_name in ckpt_list:
        adapter_params_bin_path = os.path.join(
            adapter_path_map[ckpt_name], "adapter_model.bin"
        )
        assert os.path.exists(adapter_params_bin_path)
        adapter_params = torch.load(adapter_params_bin_path)
        if adapter_param_keys is None:
            adapter_param_keys = adapter_params.keys()
        else:
            assert adapter_param_keys == adapter_params.keys()
        for k in adapter_param_keys:
            p = adapter_params[k]
            if k not in avg_params_dict:
                avg_params_dict[k] = p.clone()
            else:
                avg_params_dict[k] += p

        del adapter_params
        torch.cuda.empty_cache()

    for k in avg_params_dict.keys():
        avg_params_dict[k].div_(len(ckpt_list))

    return avg_params_dict


def add_to_json_file(result: dict, file_path: str):
    """write result to json file and check if the result is different from old result"""

    old_result = json.load(open(file_path)) if os.path.exists(file_path) else {}

    for k, v in result.items():
        if k in old_result:
            if old_result[k] != v:
                print(f"Warning: {k} is different from old result.")
                json.dump(old_result, open(file_path + ".old", "w"), indent=4)
                print(f"Old result is saved to {file_path}.old")
        else:
            old_result[k] = v

    json.dump(old_result, open(file_path, "w"), indent=4)


if __name__ == "__main__":
    config = Config().parse_args()
    os.makedirs(config.save_result_dir, exist_ok=True)
    model, tokenizer, adapter_path_map = load_model(config)

    # valid_data
    print(" Valid data ".center(80, "="))

    valid_data = load_data(config.data_dir, config.corpus, split="valid")
    valid_result_dir = os.path.join(config.save_result_dir, "valid")
    valid_results = test(
        valid_data, model, tokenizer, adapter_path_map, config, valid_result_dir
    )
    print(json.dumps(valid_results, indent=4))

    # test_data
    print(" Test data ".center(80, "="))
    test_data = load_data(config.data_dir, config.corpus, split="test")

    test_result_dir = os.path.join(config.save_result_dir, "test")
    if not config.test_all_checkpoints:
        best_and_avg_adapter_path_map = {
            subtask: {
                f"{subtask}_checkpoint-best": path_map[f"{subtask}_checkpoint-best"],
                f"{subtask}_checkpoint-avg": path_map[f"{subtask}_checkpoint-avg"],
            }
            for subtask, path_map in adapter_path_map.items()
        }
        adapter_path_map = best_and_avg_adapter_path_map

    test_results = test(
        test_data, model, tokenizer, adapter_path_map, config, test_result_dir
    )
    print(json.dumps(test_results, indent=4))
