# You might need the following imports. Feel free to change it if you opt for different libraries.
import os
import glob as globmod
from typing import Any, Self
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain_core.documents import Document

from helpers.documents import load_documents, split_documents
from helpers.embeddings import build_index


# Default configs
DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_TOP_K = 4


def _parse_int_setting(name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    return parsed


def resolve_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolves runtime configuration with defaults and typed settings."""
    config = config or {}

    resolved = {
        "api_key": config.get("api_key", None),
        "base_url": config.get("base_url", None),
        "model": config.get("model", DEFAULT_LLM_MODEL),
        "embedding_model": config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        "top_k": _parse_int_setting(
            "TOP_K",
            config.get("top_k", DEFAULT_TOP_K),
        ),
        "chunk_size": _parse_int_setting(
            "CHUNK_SIZE",
            config.get("chunk_size", DEFAULT_CHUNK_SIZE),
        ),
        "chunk_overlap": _parse_int_setting(
            "CHUNK_OVERLAP",
            config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
        ),
        "data_dir": config.get("data_dir", DEFAULT_DATA_DIR),
    }

    if resolved["top_k"] <= 0:
        raise ValueError("TOP_K must be > 0")
    if resolved["chunk_size"] <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")
    if resolved["chunk_overlap"] < 0:
        raise ValueError("CHUNK_OVERLAP must be >= 0")
    if resolved["chunk_overlap"] >= resolved["chunk_size"]:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return resolved

def retrieve(
        query: str,
        index: faiss.IndexFlatIP,
        model: SentenceTransformer,
        chunks: list[Document],
        k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Gets the most relevant chunks for a query.

    Results are ordered by similarity and include the chunk text, similarity
    score, and metadata for each matching chunk.
    """
    query_embedding = model.encode(query, convert_to_numpy=True)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    scores, indices = index.search(query_embedding, k)
    results=[]
    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[idx]
        results.append({
            "text": chunk.page_content,
            "score": float(score),
            "metadata": chunk.metadata,
        })
    return results


SYSTEM_PROMPT = """You are a helpful personal digital assistant that has access to a user's personal documents including emails, notes, SMS messages, and calendar events.

Instructions:
1. Answer questions based ONLY on the provided context from the documents.
2. For follow-up questions, maintain continuity with previous answers in the conversation history.
3. When a question refers to something mentioned earlier (e.g., "this", "that", "it"), explicitly connect it to the previous context.
4. If information conflicts between documents (e.g., different times for same event), mention both and ask for clarification.
5. Always cite which document type (EMAIL, SMS, NOTES, CALENDAR) the information comes from.
6. If you cannot find relevant information, explicitly state what you looked for and suggest rephrasing the question.
7. Be concise and factual - avoid speculation or assumptions beyond the provided context.
8. Format dates and times consistently (include timezone if available).
"""


class Assistant:
    """Stateful RAG assistant.

    The assistant owns the pipeline components, resolved configuration, and
    conversation history. Questions are answered with retrieved document context
    and the configured chat model.
    """

    def __init__(
            self,
            index: faiss.IndexFlatIP,
            model: SentenceTransformer,
            chunks: list[Document],
            client: OpenAI,
            config: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.model = model
        self.chunks = chunks
        self.client = client
        self.config = resolve_config(config)
        self.llm_model = self.config["model"]
        self.top_k = self.config["top_k"]
        self.history: list[dict[str, str]] = []

    def ask(self, question: str, k: int | None = None) -> str:
        """Generates an answer from the retrieved context and conversation history.

        The current question is combined with relevant document chunks, previous
        conversation messages, and the system prompt. The assistant response is
        appended to history alongside the user message.
        """
        k = k or self.top_k
        retrieved_chunks = retrieve(question, self.index, self.model, self.chunks, k)

        if not retrieved_chunks:
            response = "Didn't find any relevant information in your documents. Can you try rephrasing or asking about something else?"
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": response})
            return response

        context = "\n\n".join([
            f"[{chunk['metadata']['type'].upper()}] {chunk['metadata']['source']}\n{chunk['text']}"
            for chunk in retrieved_chunks
        ])

        messages = [
            {"role": "system",
             "content": "You are a helpful assistant. Answer questions based on the provided context. Always refer back to previous context when answering follow-up questions."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        pass

    def clear_history(self) -> None:
        """Empties the conversation history."""
        self.history.clear()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> Self:
        """Initializes the components required by the assistant and instantiates it

        The pipeline includes resolved configuration, loaded documents, chunked
        documents, an embedding model, a FAISS index, and an OpenAI-compatible
        client.
        """
        resolved_config = resolve_config(config)

        print("Loading documents...")
        docs = load_documents(resolved_config["data_dir"])
        print(f"  Loaded {len(docs)} documents")

        print("Splitting into chunks...")
        chunks = split_documents(
            docs,
            chunk_size=resolved_config["chunk_size"],
            chunk_overlap=resolved_config["chunk_overlap"],
        )
        print(f"  Created {len(chunks)} chunks")

        embedding_model = SentenceTransformer(resolved_config["embedding_model"])

        print("Building FAISS index...")
        index = build_index(chunks, embedding_model)
        print(f"  Indexed {index.ntotal} vectors (dim={index.d})")

        client_kwargs = {}
        if resolved_config["api_key"]:
            client_kwargs["api_key"] = resolved_config["api_key"]
        if resolved_config["base_url"]:
            client_kwargs["base_url"] = resolved_config["base_url"]
        client = OpenAI(**client_kwargs)

        print("Ready!\n")
        return cls(index, embedding_model, chunks, client, resolved_config)
