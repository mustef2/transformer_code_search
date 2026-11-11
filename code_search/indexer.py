import numpy as np

def build_index(documents: list[dict], model) -> np.ndarray:
    """Encode code units into normalized Transformer embeddings."""
    if not documents:
        return np.empty((0, 0), dtype=np.float32)
    return model.encode([d["search_text"] for d in documents],
                        convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=False)
