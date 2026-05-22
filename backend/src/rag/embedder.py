from typing import List
from fastembed import TextEmbedding, SparseTextEmbedding


class ComplianceEmbedder:
    def __init__(self):
        print("Loading Dense Embedding Model...")
        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        print("Loading Sparse Embedding Model...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def embed_text(self, text: str) -> List[float]:
        vector = list(self.dense_model.embed([text]))[0]
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = list(self.dense_model.embed(texts))
        return [v.tolist() for v in vectors]

    def embed_sparse_text(self, text: str):
        sparse_vector = list(self.sparse_model.embed([text]))[0]
        return {
            "indices": sparse_vector.indices.tolist(),
            "values": sparse_vector.values.tolist(),
        }

    def embed_sparse_batch(self, texts: List[str]):
        sparse_vectors = list(self.sparse_model.embed(texts))
        return [
            {
                "indices": vec.indices.tolist(),
                "values": vec.values.tolist(),
            }
            for vec in sparse_vectors
        ]