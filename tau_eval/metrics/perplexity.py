import evaluate


# Loading evaluation metric
perplexity = evaluate.load("perplexity", module_type="metric")


def compute_perplexity(output_texts: str | list[str], model_id: str = "gpt2") -> dict[str, list[float]]:
    """
    Computes perplexity scores for a list of output texts.

    Args:
        output_texts: A string or list of output text strings.
        model_id: HuggingFace model to use

    Returns:
        A dictionary containing perplexity scores for each input text.
    """
    if not isinstance(output_texts, list):
        output_texts = [output_texts]

    result = perplexity.compute(predictions=output_texts, model_id=model_id)

    return result
