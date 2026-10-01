from sentence_transformers import SentenceTransformer

from paperlens.config import settings

model = SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = model.encode(texts, batch_size=16, convert_to_numpy=True)
    return vectors.tolist()
