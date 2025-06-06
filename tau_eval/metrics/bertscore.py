import evaluate


# Loading evaluation metric
bert = evaluate.load("bertscore")


def compute_bertscore(
    input_texts: str | list[str],
    output_texts: str | list[str],
    model_id: str = "distilbert-base-uncased",
) -> dict[str, list[float]]:
    """
    Computes BERTScore for a list of input and output text pairs.

    Args:
        input_texts: A string or a list of input text strings.
        output_texts: A string or a list of output text strings.
        model_id: Bert specification, HuggingFace model to use.

    Returns:
        A dictionary containing BERTScore scores for each input-output pair.
        The dictionary will contain keys "precision", "recall", and "f1".
    """
    if not isinstance(input_texts, list):
        input_texts = [input_texts]
    if not isinstance(output_texts, list):
        output_texts = [output_texts]
    assert len(input_texts) == len(output_texts), "inputs are different lengths"

    return bert.compute(
        predictions=output_texts,
        references=input_texts,
        model_type=model_id,
    )
