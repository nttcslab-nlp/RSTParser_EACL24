from typing import Literal

from peft import PeftModel
from transformers import PreTrainedTokenizer

from data.relation import get_relation_labels
from data.tree import AttachTree

from ..generate_train_examples import (
    generate_nuc_example,
    generate_rel_example,
    generate_rel_with_nuc_example,
    generate_span_example,
)
from ..parse_utils import DEFAULT_LABEL, NUCLEUS_LABELS, generate_answer
from .shift_reduce_state import ShiftReduceState


def build_tree_by_shift_reduce(
    edus: list[str],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    rel_type: Literal["rel", "nuc_rel", "rel_with_nuc"] = "rel",
    with_tree: bool = False,
    prune_type: Literal["none", "nuc", "edge"] = "none",
    corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
) -> AttachTree:
    # shift reduce state
    state = ShiftReduceState(len(edus))

    # parse with model
    while not state.is_end():
        if len(state.allowed_actions()) == 2:
            # predict action with model
            act = predict_action(
                state,
                edus,
                model,
                tokenizer,
                with_tree=with_tree,
                prune_type=prune_type,
            )
            assert act in {"shift", "reduce"}, act
        elif state.is_allowed_action("shift"):
            act = "shift"
        elif state.is_allowed_action("reduce"):
            act = "reduce"
        else:
            raise AssertionError

        if act == "shift":
            nuc, rel = None, None
        else:
            # predict label with model
            nuc, rel = predict_label(
                state,
                edus,
                model,
                tokenizer,
                rel_type=rel_type,
                with_tree=with_tree,
                prune_type=prune_type,
                corpus=corpus,
            )
            assert nuc in NUCLEUS_LABELS, nuc
            assert rel in get_relation_labels(corpus), rel

        # update stack and queue
        state.operate(act, nuc, rel)

    tree = state.get_tree()
    return tree


def predict_action(
    state: ShiftReduceState,
    edus: list[str],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    with_tree: bool = False,
    prune_type: Literal["none", "nuc", "edge"] = "none",
) -> Literal["shift", "reduce"]:
    adapter_name = "span"
    if with_tree:
        adapter_name += "_with_tree"
    if prune_type == "nuc":
        adapter_name += "_prune_nuc"
    model.set_adapter(adapter_name)

    span_example = generate_span_example(
        stack1=state.stack[-1],
        stack2=state.stack[-2],
        queue1=state.queue[-1],
        action="<pad>",
        edus=edus,
        span_type="tree" if with_tree else "text",
        prune_type=prune_type,
    )

    action = generate_answer(span_example["input"], model, tokenizer, max_new_tokens=5)

    if action not in {"shift", "reduce"}:
        print(f"Invalid action `{action}`. Use `{DEFAULT_LABEL['span']}` instead.")
        action = DEFAULT_LABEL["span"]

    return action


def predict_label(
    state: ShiftReduceState,
    edus: list[str],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    rel_type: Literal["rel", "nuc_rel", "rel_with_nuc"] = "rel",
    with_tree: bool = False,
    prune_type: Literal["none", "nuc", "edge"] = "none",
    corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
) -> tuple[str, str]:
    assert rel_type in {"rel", "nuc_rel", "rel_with_nuc"}

    if rel_type == "nuc_rel":
        raise NotImplementedError
    nuc = predict_nuc(
        state, edus, model, tokenizer, with_tree=with_tree, prune_type=prune_type
    )
    if rel_type == "rel":
        rel = predict_rel(
            state,
            edus,
            model,
            tokenizer,
            with_tree=with_tree,
            prune_type=prune_type,
            corpus=corpus,
        )
    else:
        rel = predict_rel_with_nuc(
            state,
            edus,
            model,
            tokenizer,
            nuc,
            with_tree=with_tree,
            prune_type=prune_type,
            corpus=corpus,
        )

    assert nuc in NUCLEUS_LABELS, nuc
    assert rel in get_relation_labels(corpus), rel

    return nuc, rel


def predict_nuc(
    state: ShiftReduceState,
    edus: list[str],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    with_tree: bool = False,
    prune_type: Literal["none", "nuc", "edge"] = "none",
) -> str:
    adapter_name = "nuc"
    if with_tree:
        adapter_name += "_with_tree"
    if prune_type == "nuc":
        adapter_name += "_prune_nuc"
    model.set_adapter(adapter_name)

    nuc_example = generate_nuc_example(
        stack1=state.stack[-1],
        stack2=state.stack[-2],
        edus=edus,
        nuc="<pad>",
        span_type="tree" if with_tree else "text",
        prune_type=prune_type,
    )
    nuc = generate_answer(nuc_example["input"], model, tokenizer, max_new_tokens=10)

    if nuc not in NUCLEUS_LABELS:
        print(f"Invalid nuc label `{nuc}`. Use `{DEFAULT_LABEL['nuc']}` instead.")
        nuc = DEFAULT_LABEL["nuc"]
    return nuc


def predict_rel(
    state: ShiftReduceState,
    edus: list[str],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    with_tree: bool = False,
    prune_type: Literal["none", "nuc", "edge"] = "none",
    corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
) -> str:
    adapter_name = "rel"
    if with_tree:
        adapter_name += "_with_tree"
    if prune_type == "nuc":
        adapter_name += "_prune_nuc"
    model.set_adapter(adapter_name)

    rel_example = generate_rel_example(
        stack1=state.stack[-1],
        stack2=state.stack[-2],
        edus=edus,
        rel="<pad>",
        span_type="tree" if with_tree else "text",
        prune_type=prune_type,
        corpus=corpus,
    )
    rel = generate_answer(rel_example["input"], model, tokenizer, max_new_tokens=10)

    rel_labels = get_relation_labels(corpus)
    if rel not in rel_labels:
        major_rel = DEFAULT_LABEL[f"rel_{corpus}"]
        print(f"Invalid rel label `{rel}`. Use `{major_rel}` instead.")
        rel = major_rel
    return rel


def predict_rel_with_nuc(
    state: ShiftReduceState,
    edus: list[str],
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    nuc: str,
    with_tree: bool = False,
    prune_type: Literal["none", "nuc", "edge"] = "none",
    corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
) -> str:
    # set adapter
    adapter_name = "rel_with_nuc"
    if with_tree:
        adapter_name += "_with_tree"
    if prune_type == "nuc":
        adapter_name += "_prune_nuc"

    model.set_adapter(adapter_name)

    rel_with_nuc_example = generate_rel_with_nuc_example(
        stack1=state.stack[-1],
        stack2=state.stack[-2],
        edus=edus,
        nuc=nuc,
        rel="<pad>",
        span_type="tree" if with_tree else "text",
        prune_type=prune_type,
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


# def predict_nuc_rel(
#     state: ShiftReduceState,
#     edus: list[str],
#     model: PeftModel,
#     tokenizer: PreTrainedTokenizer,
#     with_tree: bool = False,
#     prune_type: Literal["none", "nuc", "edge"] = "none",
#     corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
# ) -> str:
#     # set adapter
#     adapter_name = "nuc_rel"
#     if with_tree:
#         adapter_name += "_with_tree"
#     if prune_type == "nuc":
#         adapter_name += "_prune_nuc"
#     model.set_adapter(adapter_name)

#     nuc_rel_example = generate_nuc_rel_example(
#         stack1=state.stack[-1],
#         stack2=state.stack[-2],
#         edus=edus,
#         label="<pad>",
#         span_type="tree" if with_tree else "text",
#         prune_type=prune_type,
#     )
#     label = generate_answer(
#         nuc_rel_example["input"], model, tokenizer, max_new_tokens=15
#     )

#     if ":" in label:
#         nuc, rel = label.split(":", maxsplit=1)
#     else:
#         nuc, rel = label, "<defect>"

#     if nuc not in NUCLEUS_LABELS:
#         print(f"Invalid nuc label `{nuc}`. Use `{NUCLEUS_LABELS[0]}` instead.")
#         nuc = NUCLEUS_LABELS[0]

#     rel_labels = get_relation_labels(corpus)
#     if rel not in rel_labels:
#         major_rel = DEFAULT_LABEL[f"rel_{corpus}"]
#         print(f"Invalid rel label `{rel}`. Use `{major_rel}` instead.")
#         rel = major_rel
#     return nuc, rel
