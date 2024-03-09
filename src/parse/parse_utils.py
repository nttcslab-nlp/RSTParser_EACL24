from peft import PeftModel
from transformers import PreTrainedTokenizer

# Most frequent labels used for invalid answer
DEFAULT_LABEL = {
    "span": "shift",
    "top_down": 0,
    "nuc": "nucleus-satellite",
    "rel_rstdt": "Elaboration",
    "rel_gum": "Elaboration",
    "rel_instrdt": "preparation:act",
}

NUCLEUS_LABELS = [
    "nucleus-satellite",
    "nucleus-nucleus",
    "satellite-nucleus",
]


def generate_answer(
    input_text: str,
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int = 10,
) -> str:
    """
    Generate answer with model.

    Args:
        input_text (str): input text
        model (PeftModel): model for parsing
        tokenizer (PreTrainedTokenizer): tokenizer for parsing
        max_new_tokens (int, optional): max new tokens. Defaults to 4000.

    Returns:
        str: answer text
    """
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **{k: v.cuda() for k, v in inputs.items()}, max_new_tokens=max_new_tokens
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert output_text.startswith(input_text), f"{input_text=}, {output_text=}"
    answer = output_text[len(input_text) :].strip()
    return answer
