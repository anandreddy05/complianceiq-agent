import os
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from src.rag.embedder import ComplianceEmbedder
from src.rag.retriever import ComplianceRetriever
from src.scraper.url_scraper import scrape_url

from ingest import main as run_ingest

load_dotenv(override=True)

app = FastAPI(title="ComplianceIQ")

# ─────────────────────────────────────────
# Init shared components
# ─────────────────────────────────────────
client = QdrantClient(path="./qdrant_storage")
embedder = ComplianceEmbedder()
retriever = ComplianceRetriever(client=client, embedder=embedder)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
)

# In-session conversation memory
conversation_history = []


# ─────────────────────────────────────────
# Request models
# ─────────────────────────────────────────
class AskRequest(BaseModel):
    query: str


class AskURLRequest(BaseModel):
    query: str
    url: str


# ─────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────
SYSTEM_PROMPT = """You are ComplianceIQ — an enterprise regulatory compliance assistant.

You answer questions about GDPR, India DPDP Act 2023, RBI IT guidelines, and OWASP security standards.

Rules:
- Always cite the specific Article, Section, or Clause number when available
- Never give legal advice — provide regulatory guidance only
- If unsure, say "consult a legal expert" explicitly
- Keep answers concise and structured
- Never make up regulations or article numbers"""


@app.on_event("startup")
async def startup():
    print("Checking knowledge base...")
    from qdrant_client import QdrantClient as QC

    check = QC(path="./qdrant_storage")
    if not check.collection_exists("compliance_documents"):
        print("Knowledge base empty — running ingestion...")
        run_ingest()
        print("Ingestion complete.")
    else:
        print("Knowledge base ready.")


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.post("/ask")
async def ask(request: AskRequest):
    docs = retriever.retrieve(request.query)

    if not docs:
        context = "No relevant compliance documents found in the knowledge base."
    else:
        context = "\n\n".join(
            [
                f"[{doc['metadata']['regulation']} — {doc['metadata']['source_file']}]\n{doc['content']}"
                for doc in docs
            ]
        )

    recent_history = conversation_history[-6:]

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *recent_history,
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {request.query}"),
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()

    conversation_history.append(HumanMessage(content=request.query))
    conversation_history.append(response)

    return {"answer": answer}


@app.post("/ask-url")
async def ask_url(request: AskURLRequest):
    scraped_text = scrape_url(request.url)

    if not scraped_text.strip():
        return {
            "answer": "Could not extract content from the provided URL. Please check the link and try again."
        }

    recent_history = conversation_history[-6:]

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *recent_history,
        HumanMessage(
            content=f"Source URL: {request.url}\n\nContent:\n{scraped_text}\n\nQuestion: {request.query}"
        ),
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()

    conversation_history.append(HumanMessage(content=request.query))
    conversation_history.append(response)

    return {"answer": answer}


@app.get("/health")
async def health():
    return {"status": "ok"}
