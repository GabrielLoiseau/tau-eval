import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def load_sbert(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cuda"
) -> SentenceTransformer:
    """
    Loads the sentence similarity model.

    Args:
        model_name: SentenceTransformers model to load.
        device: The device to load the model onto ("cuda" or "cpu").

    Returns:
        The loaded SentenceTransformer model.
    """
    return SentenceTransformer(model_name, device=device)


def compute_sbert(
    input_texts: str | list[str],
    output_texts: str | list[str],
    sim_model: SentenceTransformer,
) -> dict[str, list[float]]:
    """
    Computes the cosine similarity between the embeddings of original and rewritten texts.

    Args:
        original: A string or a list of original texts.
        rewrites: A string or a list of rewritten texts.
        sim_model: The loaded SentenceTransformer model.

    Returns:
        A dictionary containing the similarity scores for each input text pair.
        The dictionary has the key "similarity" with a list of float values.
    """
    if not isinstance(input_texts, list):
        input_texts = [input_texts]
    if not isinstance(output_texts, list):
        output_texts = [output_texts]
    assert len(input_texts) == len(output_texts), "inputs are different lengths"

    outputs = []
    embedding_orig: torch.Tensor = sim_model.encode(input_texts, convert_to_tensor=True, show_progress_bar=False)
    embedding_rew: torch.Tensor = sim_model.encode(output_texts, convert_to_tensor=True, show_progress_bar=False)

    for orig, new in zip(embedding_orig, embedding_rew):
        outputs.append(cos_sim(orig, new).item())

    return {"sbert": outputs}
