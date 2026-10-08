from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def embed_query(query: str) -> list[float]:
    embedding = model.encode(query)
    return embedding.tolist()
