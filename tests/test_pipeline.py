from transcript import get_transcript
from rag.chunking import chunk_text
from rag.embeddings import create_embeddings

url = input("Enter YouTube URL: ")

data = get_transcript(url)

print(f"\nTranscript Length: {len(data['text'])}")

chunks = chunk_text(data["text"])

print(f"Chunks Created: {len(chunks)}")

embeddings = create_embeddings(chunks)

print(f"Embedding Shape: {embeddings.shape}")

print("\nPipeline Success!")