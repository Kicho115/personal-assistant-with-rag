# You might need the following imports. Feel free to change it if you opt for different libraries.
import os
import glob as globmod
import re
from typing import Any, Self
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
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
DEFAULT_OVERFETCH_MULTIPLIER = 5

TAG_TO_DOC_TYPE = {
    "email": "emails",
    "notes": "notes",
    "sms": "sms",
    "calendar": "calendar",
}

_TAG_PATTERN = re.compile(
    r"/(?P<tag>email|notes|sms|calendar)\b",
    re.IGNORECASE,
)


def parse_query_tags(question: str) -> tuple[str, set[str] | None]:
    """Extract document-type tags from a question and return a clean search query.

    Tags like /calendar map to metadata ``type`` values (e.g. ``calendar``).
    When no tags are present, the original question is returned and the filter is None.
    """
    doc_types: set[str] = set()
    for match in _TAG_PATTERN.finditer(question):
        key = match.group("tag").lower()
        doc_types.add(TAG_TO_DOC_TYPE[key])

    clean_query = _TAG_PATTERN.sub("", question)
    clean_query = re.sub(r"\s+", " ", clean_query).strip()

    if not doc_types:
        return question, None

    if not clean_query:
        clean_query = " ".join(sorted(doc_types))

    return clean_query, doc_types


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
        "api_key": config.get("api_key") or None,
        "base_url": config.get("base_url") or None,
        "model": config.get("model") or DEFAULT_LLM_MODEL,
        "embedding_model": config.get("embedding_model") or DEFAULT_EMBEDDING_MODEL,
        "top_k": _parse_int_setting(
            "TOP_K",
            config.get("top_k") or DEFAULT_TOP_K,
        ),
        "chunk_size": _parse_int_setting(
            "CHUNK_SIZE",
            config.get("chunk_size") or DEFAULT_CHUNK_SIZE,
        ),
        "chunk_overlap": _parse_int_setting(
            "CHUNK_OVERLAP",
            config.get("chunk_overlap") or DEFAULT_CHUNK_OVERLAP,
        ),
        "data_dir": config.get("data_dir") or DEFAULT_DATA_DIR,
        "overfetch_multiplier": _parse_int_setting(
            "OVERFETCH_MULTIPLIER",
            config.get("overfetch_multiplier") or DEFAULT_OVERFETCH_MULTIPLIER,
        ),
    }

    if resolved["top_k"] <= 0:
        raise ValueError("TOP_K must be > 0")
    if resolved["chunk_size"] <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")
    if resolved["chunk_overlap"] < 0:
        raise ValueError("CHUNK_OVERLAP must be >= 0")
    if resolved["chunk_overlap"] >= resolved["chunk_size"]:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    if resolved["overfetch_multiplier"] <= 0:
        raise ValueError("OVERFETCH_MULTIPLIER must be > 0")

    return resolved

def retrieve(
        query: str,
        index: faiss.IndexFlatIP,
        model: SentenceTransformer,
        chunks: list[Document],
        k: int = DEFAULT_TOP_K,
        doc_types: set[str] | None = None,
        overfetch_multiplier: int = DEFAULT_OVERFETCH_MULTIPLIER,
) -> list[dict]:
    """Gets the most relevant chunks for a query.

    Results are ordered by similarity and include the chunk text, similarity
    score, and metadata for each matching chunk.

    When ``doc_types`` is set, overfetching retrieves more candidates from the
    index and keeps only chunks whose metadata ``type`` is in that set.
    """
    fetch_k = k
    if doc_types:
        fetch_k = min(k * overfetch_multiplier, index.ntotal)

    query_embedding = model.encode(query, convert_to_numpy=True)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    scores, indices = index.search(query_embedding, fetch_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        if doc_types and chunk.metadata.get("type") not in doc_types:
            continue
        results.append({
            "text": chunk.page_content,
            "score": float(score),
            "metadata": chunk.metadata,
        })
        if len(results) >= k:
            break
    return results


def expand_query(question: str, client: OpenAI, model: str) -> list[str]:
    """Generates alternative phrasings of a question for better semantic search."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"""Generate 3 alternative phrasings of this question for semantic search. Return only the questions, one per line. Do not use numbers or bullet points.
            Question: {question}"""
        }],
        temperature=0.7,
    )
    raw_text = response.choices[0].message.content.strip()

    variants = [
        line.lstrip("0123456789.-* ")
        for line in raw_text.split("\n")
        if line.strip() != ""
    ]

    return [question] + variants


def rerank(query: str, results: list[dict], reranker: CrossEncoder, top_n: int) -> list[dict]:
    """Re-ranks retrieved chunks using a cross-encoder model."""
    if not results:
        return results

    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:top_n]]


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
        self.overfetch_multiplier = self.config["overfetch_multiplier"]
        self.history: list[dict[str, str]] = []
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def ask(self, question: str, k: int | None = None) -> str:
        """Generates an answer from the retrieved context and conversation history.

        The current question is combined with relevant document chunks, previous
        conversation messages, and the system prompt. The assistant response is
        appended to history alongside the user message.
        """
        k = k or self.top_k
        search_query, doc_types = parse_query_tags(question)

        augmented_query = question
        if len(self.history) >= 2:
            last_user_message = self.history[-2]["content"]
            augmented_query = f"{last_user_message} {question}"

        query_variants = expand_query(augmented_query, self.client, self.llm_model)

        all_retrieved = {}
        for variant in query_variants:
            variant_search_query, _ = parse_query_tags(variant)
            chunks = retrieve(
                variant_search_query,
                self.index,
                self.model,
                self.chunks,
                k * self.overfetch_multiplier,
                doc_types=doc_types,
                overfetch_multiplier=self.overfetch_multiplier,
            )
            for chunk in chunks:
                chunk_id = chunk["metadata"]["source"]
                if chunk_id not in all_retrieved or chunk["score"] > all_retrieved[chunk_id]["score"]:
                    all_retrieved[chunk_id] = chunk

        retrieved_chunks = list(all_retrieved.values())[:k * self.overfetch_multiplier]

        if retrieved_chunks:
            retrieved_chunks = rerank(question, retrieved_chunks, self.reranker, k)

        if not retrieved_chunks:
            if doc_types:
                types_label = ", ".join(sorted(t.upper() for t in doc_types))
                response = (
                    f"Didn't find any relevant information in your {types_label} documents. "
                    "Can you try rephrasing or asking about something else?"
                )
            else:
                response = (
                    "Didn't find any relevant information in your documents. "
                    "Can you try rephrasing or asking about something else?"
                )
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": response})
            return response

        context = "\n\n".join([
            f"[{chunk['metadata']['type'].upper()}] {chunk['metadata']['source']}\n{chunk['text']}"
            for chunk in retrieved_chunks
        ])

        question_line = question
        if doc_types:
            types_label = ", ".join(sorted(t.upper() for t in doc_types))
            question_line = f"{question} (search limited to: {types_label})"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question_line}"},
        ]

        if self.history:
            messages = messages[:1] + self.history + messages[1:]

        response_obj = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.3,
        )
        response = response_obj.choices[0].message.content

        reference_files = set()
        for chunk in retrieved_chunks:
            reference_files.add(chunk['metadata']['source'])

        if reference_files:
            references = "\n".join([f"  • {file}" for file in sorted(reference_files)])
            response += f"\n\n*Reference:*\n{references}"

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": response})

        return response

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
