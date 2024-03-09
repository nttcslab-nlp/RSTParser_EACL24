from collections import Counter
from typing import Literal

from torchtext.vocab import vocab

from .tree import RSTTree

RSTDT_RELATION_LABELS = [
    "Elaboration",
    "Attribution",
    "Joint",
    "Same-unit",
    "Contrast",
    "Explanation",
    "Background",
    "Cause",
    "Enablement",
    "Evaluation",
    "Temporal",
    "Condition",
    "Comparison",
    "Topic-Change",
    "Summary",
    "Manner-Means",
    "Textual-organization",
    "Topic-Comment",
]


INSTRDT_RELATION_LABELS = [
    "preparation:act",
    "joint",
    "general:specific",
    "criterion:act",
    "goal:act",
    "act:goal",
    "textualorganization",
    "topic-change?",
    "step1:step2",
    "disjunction",
    "contrast1:contrast2",
    "co-temp1:co-temp2",
    "act:reason",
    "act:criterion",
    "cause:effect",
    "comparision",
    "reason:act",
    "act:preparation",
    "situation:circumstance",
    "same-unit",
    "object:attribute",
    "effect:cause",
    "prescribe-act:wrong-act",
    "indeterminate",
    "specific:general",
    "before:after",
    "set:member",
    "situation:obstacle",
    "wrong-act:prescribe-act",
    "act:constraint",
    "circumstance:situation",
    "act:side-effect",
    "obstacle:situation",
    "after:before",
    "side-effect:act",
    "wrong-act:criterion",
    "attribute:object",
    "criterion:wrong-act",
    "constraint:act",
]


def get_relation_labels(corpus: Literal["rstdt", "instrdt", "gum"]):
    if corpus in ["rstdt", "gum"]:
        return RSTDT_RELATION_LABELS
    elif corpus == "instrdt":
        return INSTRDT_RELATION_LABELS
    else:
        raise ValueError


def get_relation_vocab(corpus: Literal["rstdt", "instrdt", "gum"]):
    return vocab(
        Counter(get_relation_labels(corpus)),
        specials=["<pad>"],
    )


RSTDT_RELATION_TABLE = {
    "ROOT": "ROOT",
    "span": "span",
    "attribution": "Attribution",
    "attribution-negative": "Attribution",
    "background": "Background",
    "circumstance": "Background",
    "cause": "Cause",
    "result": "Cause",
    "cause-result": "Cause",
    "consequence": "Cause",
    "comparison": "Comparison",
    "preference": "Comparison",
    "analogy": "Comparison",
    "proportion": "Comparison",
    "condition": "Condition",
    "hypothetical": "Condition",
    "contingency": "Condition",
    "otherwise": "Condition",
    "contrast": "Contrast",
    "concession": "Contrast",
    "antithesis": "Contrast",
    "elaboration-additional": "Elaboration",
    "elaboration-general-specific": "Elaboration",
    "elaboration-part-whole": "Elaboration",
    "elaboration-process-step": "Elaboration",
    "elaboration-object-attribute": "Elaboration",
    "elaboration-set-member": "Elaboration",
    "example": "Elaboration",
    "definition": "Elaboration",
    "enablement": "Enablement",
    "purpose": "Enablement",
    "evaluation": "Evaluation",
    "interpretation": "Evaluation",
    "conclusion": "Evaluation",
    "comment": "Evaluation",
    "evidence": "Explanation",
    "explanation-argumentative": "Explanation",
    "reason": "Explanation",
    "list": "Joint",
    "disjunction": "Joint",
    "manner": "Manner-Means",
    "means": "Manner-Means",
    "problem-solution": "Topic-Comment",
    "question-answer": "Topic-Comment",
    "statement-response": "Topic-Comment",
    "topic-comment": "Topic-Comment",
    "comment-topic": "Topic-Comment",
    "rhetorical-question": "Topic-Comment",
    "summary": "Summary",
    "restatement": "Summary",
    "temporal-before": "Temporal",
    "temporal-after": "Temporal",
    "temporal-same-time": "Temporal",
    "sequence": "Temporal",
    "inverted-sequence": "Temporal",
    "topic-shift": "Topic-Change",
    "topic-drift": "Topic-Change",
    "textualorganization": "Textual-organization",
    "same-unit": "Same-unit",
    # lower -> captial
    "elaboration": "Elaboration",
    "joint": "Joint",
    "same-unit": "Same-unit",
    "explanation": "Explanation",
    "temporal": "Temporal",
    "topic-change": "Topic-Change",
    "manner-means": "Manner-Means",
    "textual-organization": "Textual-organization",
}


def re_categorize(tree: RSTTree):
    def helper(node):
        if not isinstance(node, RSTTree):
            return node

        label = node.label()
        if label not in ["ROOT", "text"]:
            nuc, rel = node.label().split(":", maxsplit=1)
            while rel[-2:] in ["-s", "-e", "-n"]:
                rel = rel[:-2]

            if rel not in RSTDT_RELATION_TABLE.values():
                assert rel.lower() in RSTDT_RELATION_TABLE.keys()
                rel = RSTDT_RELATION_TABLE[rel.lower()]
            label = ":".join([nuc, rel])

        return RSTTree(label, [helper(child) for child in node])

    assert isinstance(tree, RSTTree)
    return helper(tree)
