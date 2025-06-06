import evaluate


# Loading evaluation metric
rouge = evaluate.load("rouge")


def compute_rouge(
    input_texts: str | list[str],
    output_texts: str | list[str],
) -> dict[str, list[float]]:
    """
    Computes ROUGE scores for a list of input and output text pairs.

    Args:
        input_texts: A list of input text strings.
        output_texts: A list of output text strings.

    Returns:
        A dictionary containing ROUGE scores for each input-output pair.
    """
    if not isinstance(input_texts, list):
        input_texts = [input_texts]  # type: ignore
    if not isinstance(output_texts, list):
        output_texts = [output_texts]  # type: ignore
    assert len(input_texts) == len(output_texts), "inputs are different lengths"

    return rouge.compute(predictions=output_texts, references=input_texts, use_aggregator=False)
