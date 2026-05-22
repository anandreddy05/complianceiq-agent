from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

from .embedder import ComplianceEmbedder


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, embedder: ComplianceEmbedder):
        print("Initializing Qdrant Vector DB.")

        self.client = client
        self.embedder = embedder
        self.collection_name = "compliance_documents"

        self._setup_collection()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

    def _setup_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="regulation",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

            print(f"Collection '{self.collection_name}' created successfully.")

    def ingest_docs(
        self,
        document_id: str,
        regulation: str,
        doc_type: str,
        source_file: str,
        markdown_text: str,
    ):
        if not markdown_text or markdown_text.strip() == "":
            return

        chunks = self.text_splitter.split_text(markdown_text)
        if not chunks:
            return

        dense_vectors = self.embedder.embed_batch(chunks)
        sparse_vectors = self.embedder.embed_sparse_batch(chunks)
        points = []

        for i, (chunk, dense_vector, sparse_vector) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors)
        ):
            point_id = hashlib.md5(f"{document_id}_{i}".encode()).hexdigest()
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vector,
                        "sparse": models.SparseVector(
                            indices=sparse_vector["indices"],
                            values=sparse_vector["values"],
                        ),
                    },
                    payload={
                        "document_id": document_id,
                        "regulation": regulation,
                        "doc_type": doc_type,
                        "source_file": source_file,
                        "content_markdown": chunk,
                        "chunk_index": i,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Ingested {len(points)} chunks from: {source_file}")
