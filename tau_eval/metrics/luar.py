"""Sentence transformers version of LUAR"""
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def load_luar(
    model_name: str = "gabrielloiseau/LUAR-MUD-sentence-transformers",
    device: str = "cuda",
) -> SentenceTransformer:
    """
    Loads the LUAR (Language Understanding and Reasoning) sentence transformer model.

    Args:
        model_name: SentenceTransformers model to load.
        device: The device to load the model onto ("cuda" or "cpu").

    Returns:
        The loaded SentenceTransformer model.
    """
    return SentenceTransformer(model_name, device=device)


def compute_luar(
    input_texts: str | list[str],
    output_texts: str | list[str],
    sim_model: SentenceTransformer,
) -> dict[str, list[float]]:
    """
    Computes LUAR scores based on cosine similarity between the embeddings of original and rewritten texts.

    Args:
        original: A string or list of original texts.
        rewrites: A string or list of rewritten texts.
        sim_model: The loaded SentenceTransformer model.

    Returns:
        A dictionary containing the LUAR scores for each pair of input texts.
        The dictionary has the key "luar" with a list of float values.
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

    return {"luar": outputs}
