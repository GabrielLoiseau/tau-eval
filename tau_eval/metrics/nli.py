import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_nli(
    model_name: str = "alisawuffles/roberta-large-wanli", device: str = "cuda"
) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Loads the NLI (Natural Language Inference) model and tokenizer.

    Args:
        model_name: HuggingFace model to load
        device: The device to load the model onto ("cuda" or "cpu").

    Returns:
        A tuple containing the loaded tokenizer and model.
    """
    # Download necessary models for evaluation
    nli_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, map_location=device)
    nli_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    return (nli_tokenizer, nli_model)


def compute_nli(
    input_texts: str | list[str],
    output_texts: str | list[str],
    nli_tokenizer: PreTrainedTokenizer,
    nli_model: PreTrainedModel,
    batch_size: int = 16,
    device: str = "cuda",
    max_length: int = 128,
) -> dict[str, list[float]]:
    """
    Computes the probability of entailment between two texts using the NLI model.

    Args:
        input_text: The premise text.
        output_text: The hypothesis text.
        nli_tokenizer: The tokenizer for the NLI model.
        nli_model: The NLI model.

    Returns:
        A dictionary containing the probability of entailment.
        The dictionary has the key "entailment" with a float value.
    """
    if not isinstance(input_texts, list):
        input_texts = [input_texts]  # type: ignore
    if not isinstance(output_texts, list):
        output_texts = [output_texts]  # type: ignore
    assert len(input_texts) == len(output_texts), "inputs are different lengths"

    entailment_prob = []

    for start_index in range(0, len(output_texts), batch_size):
        end_index = min(start_index + batch_size, len(output_texts))
        input_batch = input_texts[start_index:end_index]
        output_batch = output_texts[start_index:end_index]

        x = nli_tokenizer(
            input_batch,
            output_batch,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            logits = nli_model.to(device)(**x).logits
        entailment_prob += logits.softmax(dim=1)[:, 1].tolist()

    return {"entailment": entailment_prob}
