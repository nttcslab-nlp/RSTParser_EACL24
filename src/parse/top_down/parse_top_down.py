from typing import Literal

from peft import PeftModel
from transformers import PreTrainedTokenizer

from data.relation import get_relation_labels
from data.tree import AttachTree, convert_leaves_to_idx

from ..generate_train_examples import (
    generate_nuc_example,
    generate_rel_example,
    generate_rel_with_nuc_example,
    generate_top_down_example,
)
from ..parse_utils import DEFAULT_LABEL, NUCLEUS_LABELS, generate_answer


def build_tree_by_top_down(
    edus: list[str],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    rel_type: Literal["rel", "nuc_rel", "rel_with_nuc"] = "rel",
    corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
) -> AttachTree:
    """build tree by top down parsing"""

    def build_tree(edus: list[str]) -> AttachTree:
        if len(edus) == 1:
            return AttachTree("text", [edus[0]])

        span1_edus, span2_edus = split_span(edus, model, tokenizer)

        nuc, rel = predict_label(
            span1_edus, span2_edus, model, tokenizer, rel_type, corpus=corpus
        )
        label = f"{nuc}:{rel}"

        return AttachTree(label, [build_tree(span1_edus), build_tree(span2_edus)])

    tree = build_tree(edus)
    convert_leaves_to_idx(tree)
    return tree


def split_span(
    edus: list[str], model: PeftModel, tokenizer: PreTrainedTokenizer
) -> tuple[list[str], list[str]]:
    """split span by top down parsing"""

    split_point_idx = 0
    if len(edus) > 2:
        # predict split point with model
        model.set_adapter("top_down")
        top_down_example = generate_top_down_example(edus, split_point=-1)
        split_point = generate_answer(
            top_down_example["input"], model, tokenizer, max_new_tokens=5
        )

        # validate split point
        if split_point.isdecimal() and 0 <= int(split_point) < len(edus) - 1:
            split_point_idx = int(split_point)
        else:
            print(
                f"Invalid span id `{split_point}` (max={len(edus)-2}). Use `0` instead."
            )

    return edus[: split_point_idx + 1], edus[split_point_idx + 1 :]


def predict_label(
    span1_edus: list[str],
    span2_edus: list[str],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    rel_type: Literal["rel", "nuc_rel", "rel_with_nuc"] = "rel",
    corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
) -> tuple[str, str]:
    """predict label by top down parsing"""

    span1_tree = build_dummy_tree(len(span1_edus))
    span2_tree = build_dummy_tree(len(span2_edus), start_edu_idx=len(span1_edus))
    concat_edus = span1_edus + span2_edus

    if rel_type != "nuc_rel":
        # predict nuc
        nuc = predict_nuc(span1_tree, span2_tree, concat_edus, model, tokenizer)

        if rel_type == "rel":
            rel = predict_rel(
                span1_tree, span2_tree, concat_edus, model, tokenizer, corpus=corpus
            )
        elif rel_type == "rel_with_nuc":
            rel = predict_rel_with_nuc(
                span1_tree,
                span2_tree,
                concat_edus,
                nuc,
                model,
                tokenizer,
                corpus=corpus,
            )
        else:
            raise ValueError(f"Invalid rel_type `{rel_type}`.")
    else:
        raise NotImplementedError

    return nuc, rel


def build_dummy_tree(n_edus: int, start_edu_idx: int = 0) -> AttachTree:
    """build dummy right branching tree"""

    tree = None
    for edu_idx in range(start_edu_idx, start_edu_idx + n_edus):
        if tree is None:
            tree = AttachTree("text", [str(edu_idx)])
        else:
            tree = AttachTree("<pad>:<pad>", [tree, AttachTree("text", [edu_idx])])

    return tree


def predict_nuc(
    span1_tree: AttachTree,
    span2_tree: AttachTree,
    concat_edus: list[str],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
) -> str:
    """predict nuc by top down parsing"""

    # predict
    model.set_adapter("nuc")
    nuc_example = generate_nuc_example(
        stack1=span2_tree, stack2=span1_tree, edus=concat_edus, nuc="<pad>"
    )
    nuc = generate_answer(nuc_example["input"], model, tokenizer, max_new_tokens=10)

    # validate
    if nuc not in NUCLEUS_LABELS:
        print(f"Invalid nuc label `{nuc}`. Use `{DEFAULT_LABEL['nuc']}` instead.")
        nuc = DEFAULT_LABEL["nuc"]
    return nuc


def predict_rel(
    span1_tree: AttachTree,
    span2_tree: AttachTree,
    concat_edus: list[str],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
) -> str:
    """predict rel by top down parsing"""

    # predict
    model.set_adapter("rel")
    rel_example = generate_rel_example(
        stack1=span2_tree,
        stack2=span1_tree,
        edus=concat_edus,
        rel="<pad>",
        corpus=corpus,
    )
    rel = generate_answer(rel_example["input"], model, tokenizer, max_new_tokens=1010)

    # validate
    rel_labels = get_relation_labels(corpus)
    if rel not in rel_labels:
        major_rel = DEFAULT_LABEL[f"rel_{corpus}"]
        print(f"Invalid rel label `{rel}`. Use `{major_rel}` instead.")
        rel = DEFAULT_LABEL["rel"]
    return rel


def predict_rel_with_nuc(
    span1_tree: AttachTree,
    span2_tree: AttachTree,
    concat_edus: list[str],
    nuc: str,
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
) -> str:
    """predict rel_with_nuc by top down parsing"""

    # predict
    model.set_adapter("rel_with_nuc")
    rel_with_nuc_example = generate_rel_with_nuc_example(
        stack1=span2_tree,
        stack2=span1_tree,
        edus=concat_edus,
        nuc=nuc,
        rel="<pad>",
        corpus=corpus,
    )
    rel = generate_answer(
        rel_with_nuc_example["input"], model, tokenizer, max_new_tokens=10
    )

    rel_labels = get_relation_labels(corpus)
    if rel not in rel_labels:
        major_rel = DEFAULT_LABEL[f"rel_{corpus}"]
        print(f"Invalid rel label `{rel}`. Use `{major_rel}` instead.")
        rel = major_rel
    return rel
