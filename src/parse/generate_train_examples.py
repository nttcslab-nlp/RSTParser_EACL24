from typing import Any, Literal

from datasets import Dataset
from tqdm import tqdm

from data.relation import get_relation_labels, get_relation_vocab, re_categorize
from data.tree import AttachTree, RSTTree

from .shift_reduce.shift_reduce_state import ShiftReduceState


def generate_train_examples(
    dataset: list[dict[str, Any]], corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt"
) -> dict[str, Dataset]:
    """
    Generate training examples for corpus.
    """

    data_types = ["span", "nuc", "rel", "rel-with-nuc"]
    # data_types = data_types + [f"{subtask}_with_tree" for subtask in data_types]
    # data_types = (
    #     data_types
    #     + [f"{subtask}_prune_nuc" for subtask in data_types]
    #     + [f"{subtask}_prune_edge" for subtask in data_types]
    data_types += ["top_down"]

    return_dict = {key: [] for key in data_types}
    for doc in tqdm(dataset):
        examples = doc2examples(doc, corpus=corpus)
        for key in examples.keys():
            assert isinstance(examples[key], list), f"{key=}, {examples[key]=}"
            return_dict[key] += examples[key]
    assert return_dict.keys() == set(data_types)

    return {key: Dataset.from_list(data_list) for key, data_list in return_dict.items()}


def get_rst_tree(
    doc: dict[str, Any], corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt"
) -> AttachTree:
    """
    Get RST tree from document.
    """

    rst_tree: RSTTree = RSTTree.fromstring(doc["rst_tree"])

    if corpus in {"rstdt", "gum"}:
        rst_tree = re_categorize(rst_tree)

    # rel_vocab
    assert RSTTree.check_relation(rst_tree, get_relation_vocab(corpus))

    rst_tree = RSTTree.binarize(rst_tree)
    attach_tree = RSTTree.convert_to_attach(rst_tree)

    # error check
    if doc["doc_id"] != "wsj_1189":
        rst_tree.set_label("ROOT")
        assert rst_tree == AttachTree.convert_to_rst(attach_tree)

    return attach_tree


def doc2examples(
    doc: dict[str, Any], corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt"
) -> dict[str, list[dict[str, Any]]]:
    """
    Generate training examples from a single document.
    """

    edus: list[str] = doc["edu_strings"]
    attach_tree = get_rst_tree(doc, corpus=corpus)

    # bottom-up
    return_dict = generate_shift_reduce_examples(attach_tree, edus, corpus=corpus)

    assert (
        len(return_dict["nuc"])
        # == len(return_dict["nuc_prune_nuc"])
        # == len(return_dict["nuc_prune_edge"])
        # == len(return_dict["nuc_with_tree"])
        # == len(return_dict["nuc_with_tree_prune_nuc"])
        # == len(return_dict["nuc_with_tree_prune_edge"])
        == len(return_dict["rel"])
        # == len(return_dict["rel_prune_nuc"])
        # == len(return_dict["rel_prune_edge"])
        # == len(return_dict["rel_with_tree"])
        # == len(return_dict["rel_with_tree_prune_nuc"])
        # == len(return_dict["rel_with_tree_prune_edge"])
        == len(return_dict["rel-with-nuc"])
        # == len(return_dict["rel-with-nuc_prune_nuc"])
        # == len(return_dict["rel-with-nuc_prune_edge"])
        # == len(return_dict["rel-with-nuc_with_tree"])
        # == len(return_dict["rel-with-nuc_with_tree_prune_nuc"])
        # == len(return_dict["rel-with-nuc_with_tree_prune_edge"])
    )

    # top-down
    return_dict["top_down"] = generate_top_down_examples(attach_tree, edus)

    return return_dict


def generate_top_down_examples(tree: AttachTree, edus: list[str]) -> dict[str, list]:
    """
    Generate top-down examples.
    """

    top_down_examples = []

    for tp in tree.treepositions():
        node = tree[tp]
        if not isinstance(node, AttachTree) or len(node) == 1:
            continue

        if len(node.leaves()) > 2:  # split point is not unique
            node_edus = [edus[int(leaf_idx)].strip() for leaf_idx in node.leaves()]
            split_point = len(node[0].leaves()) - 1

            example = generate_top_down_example(node_edus, split_point)
            top_down_examples.append(example)

    return top_down_examples


def generate_top_down_example(
    edus: list[str], split_point: int = -1
) -> dict[str, list]:
    """
    Generate top-down example.

    Args:
        edus (list[str]): list of edu strings
        split_point (int, optional): split point. Defaults to -1.
    """

    if split_point != -1:
        assert 0 <= split_point < len(edus) - 1

    span_text = edus[0]
    for i, edu_text in enumerate(edus[1:]):
        span_text += f" [{i}] {edu_text.strip()}"

    input_lines = [
        f"Input: {span_text}",
        f"Split point (0–{len(edus)-2}): ",
    ]
    input_text = "\n".join(input_lines)
    output_text = str(split_point)
    return {"input": input_text, "output": output_text}


def generate_shift_reduce_examples(
    tree: AttachTree,
    edus: list[str],
    corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
) -> dict[str, dict]:
    data_type = ["span", "nuc", "rel", "rel-with-nuc"]
    data_type = data_type + [f"{subtask}_with_tree" for subtask in data_type]

    data_type = (
        data_type
        + [f"{subtask}_prune_nuc" for subtask in data_type]
        + [f"{subtask}_prune_edge" for subtask in data_type]
    )

    examples = {k: [] for k in data_type}

    state = ShiftReduceState(len(tree.leaves()))
    node_stack = []

    for tp in tree.treepositions("postorder"):
        node = tree[tp]
        if not isinstance(node, AttachTree):
            continue

        s1, s2, q1 = state.get_state()

        label = node.label()

        if len(node) == 1:
            act, nuc, rel = "shift", "<pad>", "<pad>"
            assert label == "text", str(node)
            assert state.is_allowed_action(act)

            if len(state.allowed_actions()) == 2:
                # add span example
                assert node_stack[-1].leaves() == [str(idx) for idx in range(*s1)]
                assert node_stack[-2].leaves() == [str(idx) for idx in range(*s2)]
                assert q1[1] == q1[0] + 1

                for span_type in ["text", "tree"]:
                    for prune_type in ["none", "nuc", "edge"]:
                        span_example = generate_span_example(
                            stack1=node_stack[-1],
                            stack2=node_stack[-2],
                            queue1=q1[0],
                            action=act,
                            edus=edus,
                            span_type=span_type,
                            prune_type=prune_type,
                        )
                        key = (
                            "span" if span_type == "text" else f"span_with_{span_type}"
                        )
                        if prune_type != "none":
                            key += f"_prune_{prune_type}"
                        examples[key].append(span_example)

            node_stack.append(node)

        elif len(node) == 2:
            act = "reduce"
            assert state.is_allowed_action(act)

            label = node.label()
            nuc, rel = label.split(":", maxsplit=1)
            s1_node = node_stack.pop()
            s2_node = node_stack.pop()
            new_node = AttachTree(label, [s2_node, s1_node])
            assert new_node == node
            node_stack.append(new_node)

            assert s1_node.leaves() == [str(idx) for idx in range(*s1)]
            assert s2_node.leaves() == [str(idx) for idx in range(*s2)]

            # add label example
            for span_type in ["text", "tree"]:
                key_suffix_span = "" if span_type == "text" else f"_with_{span_type}"
                for prune_type in ["none", "nuc", "edge"]:
                    key_suffix = key_suffix_span
                    if prune_type != "none":
                        key_suffix += f"_prune_{prune_type}"

                    nuc_example = generate_nuc_example(
                        s1_node,
                        s2_node,
                        edus,
                        nuc,
                        span_type=span_type,
                        prune_type=prune_type,
                    )
                    examples[f"nuc{key_suffix}"].append(nuc_example)

                    rel_example = generate_rel_example(
                        s1_node,
                        s2_node,
                        edus,
                        rel,
                        span_type=span_type,
                        prune_type=prune_type,
                        corpus=corpus,
                    )
                    examples[f"rel{key_suffix}"].append(rel_example)

                    rel_with_nuc_example = generate_rel_with_nuc_example(
                        s1_node,
                        s2_node,
                        edus,
                        nuc,
                        rel,
                        span_type=span_type,
                        prune_type=prune_type,
                        corpus=corpus,
                    )
                    examples[f"rel-with-nuc{key_suffix}"].append(rel_with_nuc_example)

            if len(state.allowed_actions()) == 2:
                assert q1[1] == q1[0] + 1
                # add span example
                for span_type in ["text", "tree"]:
                    for prune_type in ["none", "nuc", "edge"]:
                        span_example = generate_span_example(
                            stack1=s1_node,
                            stack2=s2_node,
                            queue1=q1[0],
                            action=act,
                            edus=edus,
                            span_type=span_type,
                            prune_type=prune_type,
                        )
                        key = (
                            "span" if span_type == "text" else f"span_with_{span_type}"
                        )
                        if prune_type != "none":
                            key += f"_prune_{prune_type}"
                        examples[key].append(span_example)
        else:
            raise ValueError("Input tree is not binarized.")

        state.operate(act, nuc, rel)

    return examples


def prune_tree(node: AttachTree, prune_type: Literal["nuc", "edge"]) -> AttachTree:
    if prune_type == "nuc":
        """
        Prune satellite node.
        """
        if not isinstance(node, AttachTree) or node.label() == "text":
            return node
        nuc = node.label().split(":", maxsplit=1)[0]

        if nuc == "nucleus-nucleus":
            return AttachTree(
                node.label(),
                [prune_tree(node[0], prune_type), prune_tree(node[1], prune_type)],
            )
        elif nuc == "nucleus-satellite":
            return prune_tree(node[0], prune_type)
        elif nuc == "satellite-nucleus":
            return prune_tree(node[1], prune_type)
        else:
            raise ValueError(f"Invalid nucleus label: {nuc}")

    elif prune_type == "edge":
        """
        Use only edge edus.
        """
        if len(node.leaves()) == 1:
            assert node.label() == "text"
            return node

        return AttachTree(
            node.label(),
            [
                AttachTree("text", [node.leaves()[0]]),
                AttachTree("text", [node.leaves()[-1]]),
            ],
        )

    else:
        raise ValueError(f"Invalid prune_type: {prune_type}")


def node2text(
    node: AttachTree,
    edus: list[str],
    return_type: Literal["text", "tree"] = "text",
    prune_type: Literal["none", "nuc", "edge"] = "none",
) -> str:
    """
    Convert a single node to text
    """

    if prune_type != "none":
        node = prune_tree(node, prune_type=prune_type)

    if return_type == "text":
        return " ".join([edus[int(leaf_idx)].strip() for leaf_idx in node.leaves()])
    elif return_type == "tree":
        text = node._pformat_flat(nodesep="", parens="[]", quotes=False)
        for leaf_idx in node.leaves():
            text = text.replace(f"text {leaf_idx}", edus[int(leaf_idx)].strip())
        return text
    else:
        raise ValueError(f"Invalid return_type: {return_type}")


def generate_span_example(
    stack1: AttachTree,
    stack2: AttachTree,
    queue1: int | str,
    action: str,
    edus: list[str],
    span_type: Literal["text", "tree"] = "text",
    prune_type: Literal["none", "nuc", "edge"] = "none",
) -> dict[str, str]:
    """
    Generate span example.
    """

    stack2_text = node2text(stack2, edus, return_type=span_type, prune_type=prune_type)
    stack1_text = node2text(stack1, edus, return_type=span_type, prune_type=prune_type)
    queue1_text = edus[int(queue1)].strip()

    input_lines = [
        f"Stack2: {stack2_text}",
        f"Stack1: {stack1_text}",
        f"Queue1: {queue1_text}",
        "Action (shift or reduce): ",
    ]

    assert action in {"shift", "reduce", "<pad>"}, action

    return {"input": "\n".join(input_lines), "output": action}


def generate_nuc_example(
    stack1: AttachTree,
    stack2: AttachTree,
    edus: list[str],
    nuc: Literal["nucleus-nucleus", "nucleus-satellite", "satellite-nucleus", "<pad>"],
    span_type: Literal["text", "tree"] = "text",
    prune_type: Literal["none", "nuc", "edge"] = "none",
) -> dict[str, str]:
    """Generate train example to predict nucleus label."""
    span2_text = node2text(stack2, edus, return_type=span_type, prune_type=prune_type)
    span1_text = node2text(stack1, edus, return_type=span_type, prune_type=prune_type)

    input_lines = [
        f"Span1: {span2_text}",
        f"Span2: {span1_text}",
        "Nucleus label (nucleus-nucleus, nucleus-satellite, or satellite-nucleus): ",
    ]

    assert nuc in {"nucleus-nucleus", "nucleus-satellite", "satellite-nucleus", "<pad>"}

    return {"input": "\n".join(input_lines), "output": nuc}


def generate_rel_example(
    stack1: AttachTree,
    stack2: AttachTree,
    edus: list[str],
    rel: str,
    span_type: Literal["text", "tree"] = "text",
    prune_type: Literal["none", "nuc", "edge"] = "none",
    corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
) -> dict[str, str]:
    """
    Generate train example to predict relation label.
    """

    span2_text = node2text(stack2, edus, return_type=span_type, prune_type=prune_type)
    span1_text = node2text(stack1, edus, return_type=span_type, prune_type=prune_type)

    rel_labels = get_relation_labels(corpus)

    input_lines = [
        f"Span1: {span2_text}",
        f"Span2: {span1_text}",
        "Relation label ({}, or {}): ".format(
            ", ".join(rel_labels[:-1]), rel_labels[-1]
        ),
    ]

    assert rel in rel_labels + ["<pad>"]

    return {"input": "\n".join(input_lines), "output": rel}


def generate_rel_with_nuc_example(
    stack1,
    stack2,
    edus,
    nuc,
    rel,
    span_type: Literal["text", "tree"] = "text",
    prune_type: Literal["none", "nuc", "edge"] = "none",
    corpus: Literal["rstdt", "instrdt", "gum"] = "rstdt",
) -> dict[str, str]:
    """
    Generate train example to predict relation label.
    """

    span2_text = node2text(stack2, edus, return_type=span_type, prune_type=prune_type)
    span1_text = node2text(stack1, edus, return_type=span_type, prune_type=prune_type)

    rel_labels = get_relation_labels(corpus=corpus)
    input_lines = [
        f"Span1: {span2_text}",
        f"Span2: {span1_text}",
        f"Nucleus label: {nuc}",
        "Relation label ({}, or {}): ".format(
            ", ".join(rel_labels[:-1]), rel_labels[-1]
        ),
    ]

    assert rel in rel_labels + ["<pad>"]

    return {"input": "\n".join(input_lines), "output": rel}
