
"""
DocChat - RAG engine
PDF retrieval + Gemini generation + page citations.
Uses Google's official google-genai SDK.
"""

import argparse
import functools
import os
import sys
import textwrap

from dotenv import load_dotenv

load_dotenv(override=True)

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DB_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
COLLECTION = "docchat"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

BACKENDS = {
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "model": os.environ.get("LLM_MODEL", "gemini-2.5-flash"),
    },
    "groq": {
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "ollama": {
        "key_env": None,
        "model": "llama3.2",
    },
}

SYSTEM_PROMPT = """You are DocChat, a document question-answering assistant.

Rules:
1. Answer ONLY using the supplied document passages.
2. If the answer is absent, say exactly:
   I don't know from these documents.
3. Cite every factual claim using passage references such as [1] or [2].
4. Never invent facts, page numbers, figures, or dates.
5. Be clear and concise.
"""

USER_TEMPLATE = """Document passages:

{context}

Question: {question}

Answer with citations:"""


def get_key(name):
    if not name:
        return None

    value = os.environ.get(name)
    if value:
        return value.strip()

    try:
        import streamlit as st
        value = st.secrets.get(name)
        if value:
            return value.strip()
    except Exception:
        pass

    return None


@functools.lru_cache(maxsize=1)
def _embedder():
    return SentenceTransformer(EMBED_MODEL)


@functools.lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    try:
        return client.get_collection(COLLECTION)
    except Exception as exc:
        raise RuntimeError(
            f"No '{COLLECTION}' collection found in {DB_DIR}. "
            "Upload and ingest a document first."
        ) from exc


def retrieve(question, k=4, source=None):
    """Retrieve the most relevant document chunks."""

    vector = _embedder().encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    kwargs = {
        "query_embeddings": [vector],
        "n_results": k
    }

    if source:
        kwargs["where"] = {"source": source}

    result = _collection().query(**kwargs)

    hits = []

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for doc, metadata, distance in zip(
        documents, metadatas, distances
    ):
        hits.append({
            "text": doc,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "?"),
            "similarity": round(1.0 - float(distance), 3)
        })

    return hits


def build_context(hits):
    """Format retrieved passages with citation numbers."""

    return "\n\n".join(
        f"[{i}] Source: {hit['source']}, "
        f"Page: {hit['page']}\n{hit['text']}"
        for i, hit in enumerate(hits, 1)
    )


def generate_gemini(question, hits, temperature=0.0):
    """Generate an answer using Google's official Gemini SDK."""

    api_key = get_key("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add your actual API key to .env."
        )

    from google import genai
    from google.genai import types

    model = os.environ.get(
        "LLM_MODEL", "gemini-3.8-flash"
    ).strip()

    client = genai.Client(api_key=api_key)

    prompt = USER_TEMPLATE.format(
        context=build_context(hits),
        question=question
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature
        )
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text.strip(), model


def generate_openai_compatible(
    question, hits, backend, temperature=0.0
):
    """Optional Groq and Ollama support."""

    from openai import OpenAI

    if backend == "groq":
        api_key = get_key("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing.")

        base_url = "https://api.groq.com/openai/v1"
        model = os.environ.get(
            "LLM_MODEL", "llama-3.3-70b-versatile"
        )

    else:
        api_key = "ollama"
        base_url = os.environ.get(
            "OLLAMA_URL", "http://localhost:11434/v1"
        )
        model = os.environ.get("LLM_MODEL", "llama3.2")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    context=build_context(hits),
                    question=question
                )
            }
        ]
    )

    return response.choices[0].message.content.strip(), model


def answer(
    question,
    k=4,
    source=None,
    temperature=0.0,
    backend=None
):
    """Complete RAG pipeline. Returns answers, hits and model."""

    hits = retrieve(question, k=k, source=source)

    if not hits:
        return {
            "answer": "No relevant documents were found.",
            "hits": [],
            "model": None
        }

    selected_backend = (
        backend or os.environ.get("LLM_BACKEND", "gemini")
    ).lower().strip()

    try:
        if selected_backend == "gemini":
            response, model = generate_gemini(
                question, hits, temperature
            )

        elif selected_backend in ("groq", "ollama"):
            response, model = generate_openai_compatible(
                question, hits, selected_backend, temperature
            )

        else:
            raise ValueError(
                f"Unsupported backend: {selected_backend}"
            )

        return {
            "answer": response,
            "hits": hits,
            "model": model
        }

    except Exception as exc:
        return {
            "answer": (
                f"AI GENERATION ERROR: {type(exc).__name__}: {exc}\n\n"
                "Document retrieval is working. Check the API key, "
                "model name, internet connection and API availability."
            ),
            "hits": hits,
            "model": None
        }


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about indexed documents."
    )

    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("--source", default=None)
    parser.add_argument("--retrieve-only", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["gemini", "groq", "ollama"],
        default=None
    )

    args = parser.parse_args()

    if args.retrieve_only:
        for i, hit in enumerate(
            retrieve(args.question, args.k, args.source), 1
        ):
            print(
                f"\n[{i}] {hit['source']} "
                f"page {hit['page']} "
                f"similarity {hit['similarity']}"
            )
            print(textwrap.fill(hit["text"][:500], 100))
        return

    result = answer(
        args.question,
        k=args.k,
        source=args.source,
        backend=args.backend
    )

    print("\n" + result["answer"])

    print("\nSources:")
    for i, hit in enumerate(result["hits"], 1):
        print(
            f"[{i}] {hit['source']} - Page {hit['page']}"
        )

    if result["model"]:
        print("\nModel:", result["model"])


if __name__ == "__main__":
    main()
