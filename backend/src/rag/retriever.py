from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed.rerank.cross_encoder import TextCrossEncoder

from .embedder import ComplianceEmbedder
from .optimizer import QueryOptimizer


class ComplianceRetriever:
    def __init__(self, client: QdrantClient, embedder: ComplianceEmbedder):
        print("Initializing Hybrid Compliance Retriever...")

        self.client = client
        self.embedder = embedder
        self.collection_name = "compliance_documents"
        self.optimizer = QueryOptimizer()

        print("Loading Cross-Encoder Reranker...")
        self.reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base")

    def retrieve(self, query: str, limit: int = 5):
        expanded_query = self.optimizer.expand_query(query)

        print("\n--- RAG PIPELINE ---")
        print(f"Original Query: {query}")
        print(f"Expanded Query: {expanded_query}\n")

        dense_vector = self.embedder.embed_text(query)
        sparse_vector_dict = self.embedder.embed_sparse_text(expanded_query)

        sparse_vector = models.SparseVector(
            indices=sparse_vector_dict["indices"],
            values=sparse_vector_dict["values"],
        )

        initial_results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=20,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20,
        )

        if not initial_results.points:
            return []

        documents = []
        chunk_texts = []

        for point in initial_results.points:
            text = point.payload.get("content_markdown", "")
            documents.append(point)
            chunk_texts.append(text)

        new_scores = list(self.reranker.rerank(query, chunk_texts))
        scored_documents = list(zip(documents, new_scores))
        scored_documents.sort(key=lambda x: x[1], reverse=True)

        retrieved_docs = []

        for best_point, score in scored_documents[:limit]:
            payload = best_point.payload
            retrieved_docs.append(
                {
                    "content": payload.get("content_markdown", ""),
                    "score": float(score),
                    "metadata": {
                        "document_id": payload.get("document_id"),
                        "regulation": payload.get("regulation"),
                        "doc_type": payload.get("doc_type"),
                        "source_file": payload.get("source_file"),
                        "chunk_index": payload.get("chunk_index"),
                    },
                }
            )

        return retrieved_docs
