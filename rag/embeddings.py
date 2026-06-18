from sentence_transformers import SentenceTransformer

# model = SentenceTransformer(
#     "BAAI/bge-small-en-v1.5"
# )

_model = None

def get_embedding_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

    return _model


def create_embeddings(chunks):
    """
    Convert text chunks into embeddings.
    """
    model = get_embedding_model()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings