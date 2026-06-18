import faiss
import numpy as np


def create_index(embeddings):
    """
    Create a FAISS index from embeddings.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        embeddings.astype(np.float32)
    )

    return index