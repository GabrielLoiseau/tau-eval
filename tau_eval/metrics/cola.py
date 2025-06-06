import numpy as np
import torch
from nltk.tokenize import sent_tokenize

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_cola(
    model_name: str = "textattack/roberta-base-CoLA", device: str = "cuda"
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads the CoLA (Corpus of Linguistic Acceptability) model and tokenizer.

    Args:
        model_name: HuggingFace model to load
        device: The device to load the model onto ("cuda" or "cpu").

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    cola_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, map_location=device)
    cola_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return (cola_tokenizer, cola_model)


def cola_score(
    text: str,
    cola_tokenizer: PreTrainedTokenizer,
    cola_model: PreTrainedModel,
    device: str = "cuda",
) -> float:
    """
    Calculates the CoLA score for a single piece of text.

    Args:
        text: The text to score.
        cola_tokenizer: The tokenizer for the CoLA model.
        cola_model: The CoLA model.
        device: The device to run the model on ("cuda" or "cpu").

    Returns:
        The CoLA score (a float between 0 and 1).
    """
    tokenize_input = cola_tokenizer.tokenize(text)
    tensor_input = torch.tensor([cola_tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
    output = cola_model(tensor_input)
    return output.logits.softmax(-1)[0][1].item()


def compute_cola(
    output_texts: str | list[str],
    cola_tokenizer: PreTrainedTokenizer,
    cola_model: PreTrainedModel,
    device: str = "cuda",
) -> dict[str, list[float]]:
    """
    Computes CoLA scores for a list of input texts.

    Args:
        output_texts: A list of text strings.
        cola_tokenizer: The tokenizer for the CoLA model.
        cola_model: The CoLA model.
        device: The device to run the model on ("cuda" or "cpu").

    Returns:
        A dictionary containing CoLA scores for each input text.
    """
    if not isinstance(output_texts, list):
        output_texts = [output_texts]

    cola_ls: list[float] = []
    for i, text in enumerate(output_texts):
        if text == "":
            continue
        # if text is too big, break it up
        sentences = sent_tokenize(text)
        cola_score_list: list[float] = []

        for sent in sentences:
            cola_score_list.append(cola_score(sent, cola_tokenizer, cola_model, device))

        cola_ls.append(np.mean(cola_score_list))

    return {"cola": cola_ls}
