import evaluate


# Loading evaluation metric
meteor = evaluate.load("meteor")


def compute_meteor(
    input_texts: str | list[str],
    output_texts: str | list[str],
    alpha: float = 0.9,
    beta: float = 3,
    gamma: float = 0.5,
) -> dict[str, list[float]]:
    """
    Computes METEOR scores for a list of input and output text pairs.

    Args:
        input_texts: A list of input text strings.
        output_texts: A list of output text strings.
        alpha: Parameter for controlling relative weights of precision and recall.
        beta: Parameter for controlling shape of penalty function.
        gamma: Relative weight of fragmentation penalty.

    Returns:
        A dictionary containing METEOR scores for each input-output pair.
    """
    if not isinstance(input_texts, list):
        input_texts = [input_texts]  # type: ignore
    if not isinstance(output_texts, list):
        output_texts = [output_texts]  # type: ignore
    assert len(input_texts) == len(output_texts), "inputs are different lengths"

    scores = []
    for input_text, output_text in zip(input_texts, output_texts):
        scores.append(
            meteor.compute(
                predictions=[output_text],
                references=[input_text],
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )["meteor"]
        )

    return {"meteor": scores}
