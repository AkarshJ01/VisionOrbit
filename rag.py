import os
import json
from typing import Optional, Dict, Any, Tuple, List

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone


# ============================================================
# CONFIG
# ============================================================

MODEL = "gpt-oss:20b"
EMBEDDING_MODEL = "nomic-embed-text:latest"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")



# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# ============================================================
# DEFAULT LLM
# ============================================================

llm = ChatOllama(
    model=MODEL
)


# ============================================================
# PINECONE
# ============================================================

pc = None
index = None
vectorstore = None
retriver = None


if PINECONE_API_KEY and PINECONE_INDEX:

    try:

        pc = Pinecone(
            api_key=PINECONE_API_KEY
        )

        index = pc.Index(
            PINECONE_INDEX
        )

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings
        )

        retriver = vectorstore.as_retriever(
            search_kwargs={
                "k": 3
            }
        )

        print(
            "[RAG] Pinecone initialized successfully."
        )

    except Exception as e:

        print(
            f"[RAG] Pinecone initialization failed: {e}"
        )

else:

    print(
        "[RAG] Pinecone environment variables not configured."
    )


# ============================================================
# RETRIEVER
# ============================================================

def get_retriever(k: int = 3):

    global vectorstore
    global retriver

    if vectorstore is None:
        return retriver

    try:

        return vectorstore.as_retriever(
            search_kwargs={
                "k": k
            }
        )

    except Exception as e:

        print(
            f"[RAG] Failed to create retriever: {e}"
        )

        return retriver


# ============================================================
# BASIC RETRIEVAL CHAIN
# ============================================================

def retrieve_documents(query: str, k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve relevant documents and formatted sources from Pinecone."""
    r = get_retriever(k=k)
    if r is None:
        return "", []

    try:
        docs = r.invoke(query)
        context = format_doc(docs)
        sources = extract_sources(docs)
        return context, sources
    except Exception as e:
        print(f"[RAG] Document retrieval error: {e}")
        return "", []


def retrival_chain(
    query: str,
    model_name: Optional[str] = None
):

    target_model = model_name or MODEL

    # --------------------------------------------------------
    # Normalize frontend Ollama model names
    #
    # Frontend may send:
    #     ollama:gpt-oss:20b
    #
    # ChatOllama expects:
    #     gpt-oss:20b
    # --------------------------------------------------------

    if target_model.startswith("ollama:"):

        target_model = target_model[
            len("ollama:"):
        ]

    print(
        f"[RAG] Using model: {target_model}"
    )

    r = retriver

    if r is None:

        target_llm = ChatOllama(
            model=target_model
        )

        response = target_llm.invoke(
            query
        )

        return response.content

    docs = r.invoke(
        query
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    message = f"""
Use the following retrieved knowledge to answer the question.

Retrieved knowledge:
{context}

Question:
{query}

Answer using only information supported by the retrieved knowledge
and the question. Do not invent metadata or facts.
"""

    target_llm = ChatOllama(
        model=target_model
    )

    response = target_llm.invoke(
        message
    )

    return response.content


# ============================================================
# RETRIEVAL WITH SOURCES
# ============================================================

def retrival_chain_with_sources(
    query: str,
    model_name: Optional[str] = None,
    k: int = 3
) -> Dict[str, Any]:

    r = get_retriever(
        k=k
    ) or retriver

    target_model = model_name or MODEL

    # --------------------------------------------------------
    # Normalize Ollama model identifier
    # --------------------------------------------------------

    if target_model.startswith("ollama:"):

        target_model = target_model[
            len("ollama:"):
        ]

    print(
        f"[RAG] Using model: {target_model}"
    )

    # --------------------------------------------------------
    # No retriever available
    # --------------------------------------------------------

    if r is None:

        try:

            fallback_llm = ChatOllama(
                model=target_model
            )

            response = fallback_llm.invoke(
                query
            )

            return {
                "reply": response.content,
                "provider_used":
                    f"Ollama ({target_model})",
                "model_used":
                    target_model,
                "rag_sources": [],
                "context": ""
            }

        except Exception as e:

            return {
                "reply":
                    f"LLM generation failed: {str(e)}",
                "provider_used":
                    "Ollama",
                "model_used":
                    target_model,
                "rag_sources": [],
                "context": ""
            }

    # --------------------------------------------------------
    # Retrieve documents
    # --------------------------------------------------------

    try:

        docs = r.invoke(
            query
        )

    except Exception as e:

        print(
            f"[RAG] Retrieval failed: {e}"
        )

        docs = []

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    sources = []

    for doc in docs:

        context_parts.append(
            doc.page_content
        )

        metadata = getattr(
            doc,
            "metadata",
            {}
        )

        sources.append(
            metadata
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # LLM prompt
    # --------------------------------------------------------

    message = f"""
You are an AI assistant analyzing satellite imagery.

Use the retrieved knowledge below to help answer the user's question.

Retrieved knowledge:
{context}

User question:
{query}

Rules:
- Separate observations from retrieved knowledge.
- Do not invent satellite metadata.
- Do not invent sensor, date, CRS, GSD, or spectral-band information.
- If something cannot be determined, explicitly say so.
- Give a concise but useful answer.
"""

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    try:

        target_llm = ChatOllama(
            model=target_model
        )

        response = target_llm.invoke(
            message
        )

        return {
            "reply":
                response.content,
            "provider_used":
                f"Ollama ({target_model})",
            "model_used":
                target_model,
            "rag_sources":
                sources,
            "context":
                context
        }

    except Exception as e:

        return {
            "reply":
                f"LLM generation failed: {str(e)}",
            "provider_used":
                "RAG",
            "model_used":
                target_model,
            "rag_sources":
                sources,
            "context":
                context
        }


# ============================================================
# SAR DETECTOR + RAG + LLM
# ============================================================

def detector_rag_chain(
    query: str,
    detector_output: dict,
    model_name: Optional[str] = None,
    k: int = 3
) -> Dict[str, Any]:

    print(
        "\n[RAG] Starting detector → RAG → LLM pipeline..."
    )

    # ========================================================
    # MODEL NORMALIZATION
    # ========================================================

    target_model = model_name or MODEL

    # Frontend can send:
    #
    #     ollama:gpt-oss:20b
    #
    # but ChatOllama needs:
    #
    #     gpt-oss:20b
    #
    if target_model.startswith("ollama:"):

        target_model = target_model[
            len("ollama:"):
        ]

    print(
        f"[RAG] model_name received: {model_name!r}"
    )

    print(
        f"[RAG] target_model: {target_model!r}"
    )

    # ========================================================
    # DETECTOR JSON
    # ========================================================

    detector_json = json.dumps(
        detector_output,
        indent=2,
        default=str
    )

    print(
        "[RAG] Detector output prepared."
    )

    # ========================================================
    # BUILD RETRIEVAL QUERY
    # ========================================================

    retrieval_query = f"""
Satellite image analysis.

User question:
{query}

Detector output:
{detector_json}

Retrieve knowledge relevant to interpreting these detections,
satellite imagery, object detection results, SAR imagery,
and possible scene characteristics.
"""

    # ========================================================
    # RETRIEVE DOCUMENTS
    # ========================================================

    r = get_retriever(
        k=k
    ) or retriver

    docs = []

    if r is not None:

        try:

            docs = r.invoke(
                retrieval_query
            )

            print(
                f"[RAG] Retrieved {len(docs)} documents."
            )

        except Exception as e:

            print(
                f"[RAG] Retrieval failed: {e}"
            )

    else:

        print(
            "[RAG] No Pinecone retriever available."
        )

    # ========================================================
    # BUILD RAG CONTEXT
    # ========================================================

    context_parts = []

    sources = []

    for doc in docs:

        page_content = getattr(
            doc,
            "page_content",
            ""
        )

        if page_content:

            context_parts.append(
                page_content
            )

        metadata = getattr(
            doc,
            "metadata",
            {}
        )

        sources.append(
            metadata
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    # ========================================================
    # LLM PROMPT
    # ========================================================

    llm_prompt = f"""
You are VisionOrbit, an AI system for satellite-image analysis.

Analyze the provided detector output together with the retrieved
RAG knowledge and answer the user's question.

IMPORTANT:
1. Clearly distinguish what is directly supported by the detector.
2. Distinguish detector results from general knowledge retrieved
   from RAG.
3. Do not invent metadata.
4. Do not assume the satellite, sensor, acquisition date, CRS,
   GSD, spectral bands, geographic location, or resolution unless
   explicitly provided.
5. If something cannot be determined from the available data,
   say that it cannot be determined.
6. Do not treat a detector class as proof of a specific real-world
   object if the detector output does not establish that.
7. Mention uncertainty when appropriate.
8. Keep the answer useful and reasonably concise.

============================================================
USER QUESTION
============================================================

{query}

============================================================
SAR DETECTOR OUTPUT
============================================================

{detector_json}

============================================================
RETRIEVED RAG KNOWLEDGE
============================================================

{context}

============================================================
ANSWER
============================================================
"""

    # ========================================================
    # CALL OLLAMA
    # ========================================================

    try:

        print(
            f"[RAG] Calling Ollama model: {target_model}"
        )

        target_llm = ChatOllama(
            model=target_model
        )

        response = target_llm.invoke(
            llm_prompt
        )

        print(
            "[RAG] LLM generation successful."
        )

        return {
            "reply":
                response.content,
            "provider_used":
                f"SAR Detection + Pinecone RAG + Ollama ({target_model})",
            "model_used":
                target_model,
            "rag_sources":
                sources,
            "context":
                context,
            "detector_output":
                detector_output
        }

    except Exception as e:

        print(
            f"[RAG ERROR] LLM generation failed: {e}"
        )

        return {
            "reply":
                "Detector inference and RAG succeeded, "
                f"but LLM generation failed: {str(e)}",
            "provider_used":
                "SAR Detection + RAG",
            "model_used":
                target_model,
            "rag_sources":
                sources,
            "context":
                context,
            "detector_output":
                detector_output
        }