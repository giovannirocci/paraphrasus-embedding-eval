from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from sentence_transformers import SentenceTransformer

# ---------- Model registry & helpers ----------

_MODEL_CACHE: Dict[str, SentenceTransformer] = {}

def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device=device)
    return _MODEL_CACHE[model_name]

def _encode(texts: List[str], model_name: str, batch_size: int = 256, normalize: bool = True) -> np.ndarray:
    model = _get_model(model_name)
    with torch.inference_mode():
        embs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,  # makes cosine == dot product and in [-1,1]
        )
    return embs.astype(np.float32, copy=False)

# ---------- API expected by the new evaluator ----------

def pair_embeddings(
    pairs: List[Tuple[str, str]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given two texts → two embeddings.
    Returns: (E1, E2) shaped (N, D) each.
    """
    if not pairs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    t1 = [a for a, _ in pairs]
    t2 = [b for _, b in pairs]

    E1 = _encode(t1, model_name=model_name, batch_size=batch_size, normalize=normalize)
    E2 = _encode(t2, model_name=model_name, batch_size=batch_size, normalize=normalize)

    return E1, E2


def cosine_scores_from_pairs(
    pairs: List[Tuple[str, str]],
    **kwargs: Any,
) -> np.ndarray:
    """
    Convenience scorer: returns cosine similarity for each pair.
    """
    E1, E2 = pair_embeddings(pairs, **kwargs)
    if E1.size == 0:
        return np.zeros((0,), dtype=np.float32)
    # normalized embeddings ⇒ cosine == rowwise dot
    return (E1 * E2).sum(axis=1)
