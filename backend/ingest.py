import os
import fitz
from qdrant_client import QdrantClient
from dotenv import load_dotenv

from src.rag.embedder import ComplianceEmbedder
from src.rag.vectorstore import QdrantVectorStore

load_dotenv(override=True)

PDF_DIR = "compliance_pdfs"

DOCUMENTS = [
    {
        "document_id": "gdpr-celex",
        "regulation": "GDPR",
        "doc_type": "regulation",
        "source_file": "CELEX_32016R0679_EN_TXT.pdf",
    },
    {
        "document_id": "gdpr-formatted",
        "regulation": "GDPR",
        "doc_type": "regulation",
        "source_file": "gdpr.pdf",
    },
    {
        "document_id": "dpdp-act-2023",
        "regulation": "DPDP",
        "doc_type": "act",
        "source_file": "dpdp.pdf",
    },
]


def extract_text_from_pdf(filename: str) -> str:
    filepath = os.path.join(PDF_DIR, filename)
    doc = fitz.open(filepath)
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text


def main():
    print("Connecting to Qdrant...")
    client = QdrantClient(path="./qdrant_storage")

    embedder = ComplianceEmbedder()
    vectorstore = QdrantVectorStore(client=client, embedder=embedder)

    for doc in DOCUMENTS:
        print(f"\nProcessing: {doc['document_id']}...")
        text = extract_text_from_pdf(doc["source_file"])

        if not text.strip():
            print(f"Warning: No text extracted from {doc['source_file']}, skipping.")
            continue

        vectorstore.ingest_docs(
            document_id=doc["document_id"],
            regulation=doc["regulation"],
            doc_type=doc["doc_type"],
            source_file=doc["source_file"],
            markdown_text=text,
        )

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
