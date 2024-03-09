from typing import Any, Literal

from peft import PeftModel
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from data.tree import AttachTree, convert_leaves_to_idx

from .generate_train_examples import get_rst_tree
from .shift_reduce.parse_shift_reduce import build_tree_by_shift_reduce
from .top_down.parse_top_down import build_tree_by_top_down


def parse_dataset(
    dataset: list[dict[str, Any]],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    parse_type: Literal["bottom_up", "top_down"] = "bottom_up",
    rel_type: Literal["rel", "nuc_rel", "rel_with_nuc"] = "rel",
    with_tree: bool = False,
    prune_type: Literal["none", "nuc", "edge"] = "none",
    corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
) -> list[dict]:
    results = {"doc_id": [], "gold_tree": [], "pred_tree": []}
    for doc in tqdm(dataset, dynamic_ncols=True, leave=False):
        # gold tree
        gold_tree = get_rst_tree(doc, corpus=corpus)

        # parse tree (pred tree) with models
        pred_tree = parse_doc(
            doc,
            model,
            tokenizer,
            parse_type=parse_type,
            rel_type=rel_type,
            with_tree=with_tree,
            prune_type=prune_type,
            corpus=corpus,
        )
        results["doc_id"].append(doc["doc_id"])
        results["gold_tree"].append(gold_tree)
        results["pred_tree"].append(pred_tree)

    return results


def parse_doc(
    doc: dict[str, Any],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    parse_type: Literal["bottom_up", "top_down"] = "bottom_up",
    rel_type: Literal["rel", "nuc_rel", "rel_with_nuc"] = "rel",
    with_tree: bool = False,
    prune_type: Literal["none", "nuc", "edge"] = "none",
    corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
) -> AttachTree:
    """Parse document with models

    Args:
        doc (dict[str, Any]): document
        models (dict[str, PreTrainedModel]): models for parsing
        tokenizer (PreTrainedTokenizer): tokenizer for parsing
    """

    edus: list[str] = [edu.strip() for edu in doc["edu_strings"]]

    if parse_type == "bottom_up":
        tree = build_tree_by_shift_reduce(
            edus,
            model,
            tokenizer,
            rel_type=rel_type,
            with_tree=with_tree,
            prune_type=prune_type,
            corpus=corpus,
        )
    elif parse_type == "top_down":
        tree = build_tree_by_top_down(
            edus, model, tokenizer, rel_type=rel_type, corpus=corpus
        )
    else:
        raise ValueError(f"Unknown parse_type: {parse_type}")

    convert_leaves_to_idx(tree)
    return tree
