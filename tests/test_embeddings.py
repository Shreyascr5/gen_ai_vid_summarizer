from rag.chunking import chunk_text
from rag.embeddings import create_embeddings

sample_text = """
Artificial Intelligence is transforming the world.
Machine Learning is a subset of AI.
Deep Learning is a subset of Machine Learning.
""" * 100

chunks = chunk_text(sample_text)

embeddings = create_embeddings(chunks)

print("Chunks:", len(chunks))
print("Embedding Shape:", embeddings.shape)