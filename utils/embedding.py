import sentence_transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
def embed(text: str) -> list[float]:
    return model.encode(text, normalize_embeddings=True).tolist()

