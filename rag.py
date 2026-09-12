import os
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
# pyrefly: ignore [missing-import]
from langchain_pinecone import PineconeVectorStore

load_dotenv()

MODEL = "gpt-oss:20b"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Initialize embeddings and LLM
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOllama(model=MODEL)

_vectorstore = None
_retriever = None


def get_vectorstore():
    """Lazy initialize and return Pinecone VectorStore."""
    global _vectorstore
    if _vectorstore is None:
        index_name = os.environ.get("INDEX_NAME", "")
        if index_name:
            _vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings
            )
    return _vectorstore


def get_retriever(k: int = 3):
    """Get Pinecone retriever with specified k top documents."""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": k})


# Global default retriever for backwards compatibility
try:
    if os.environ.get("INDEX_NAME"):
        vectorstore = PineconeVectorStore(
            index_name=os.environ["INDEX_NAME"],
            embedding=embeddings
        )
        retriver = vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        vectorstore = None
        retriver = None
except Exception as e:
    print(f"[RAG] Warning initializing vectorstore: {e}")
    vectorstore = None
    retriver = None

prompt_template = ChatPromptTemplate.from_template("""
You are an expert AI assistant specializing in satellite remote sensing, Synthetic Aperture Radar (SAR), and earth observation.
Answer the user's question thoroughly and accurately based ONLY on the provided context from the knowledge base.

Guidelines:
- Structure your answer clearly with Markdown headings, bullet points, and numbered lists where helpful.
- Highlight key terminology in **bold**.
- Explain technical mechanisms clearly (e.g. wavelength, polarization, penetration, active microwave vs passive optical).
- If the context contains specific formulas, applications, or examples, include them.
- If the context does not contain enough information to answer fully, state what is known from the context and what is missing.

Context:
{context}

Question: {query}

Detailed Answer:
""")


def format_doc(docs) -> str:
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def extract_sources(docs) -> List[Dict[str, Any]]:
    """Extract structured source metadata and snippets from documents."""
    sources = []
    for i, doc in enumerate(docs):
        source_path = doc.metadata.get("source", "Knowledge Base Document")
        filename = os.path.basename(source_path)
        page = doc.metadata.get("page", 0)
        snippet = doc.page_content[:250].strip() + ("..." if len(doc.page_content) > 250 else "")
        
        sources.append({
            "id": i + 1,
            "filename": filename,
            "source_path": source_path,
            "page": page + 1 if isinstance(page, int) else page,
            "snippet": snippet,
            "content": doc.page_content
        })
    return sources


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


def retrival_chain(query: str, model_name: Optional[str] = None) -> str:
    """Core retrieval chain returning the final string output."""
    r = get_retriever(k=3) or retriver
    if r is None:
        return "Pinecone index is not configured or available."

    # Step 1: Retrieve relevant documents
    docs = r.invoke(query)
    print(f"Retrieved {len(docs)} documents")

    # Step 2: Format documents into context
    context = format_doc(docs)

    # Step 3: Create prompt
    message = prompt_template.format_messages(
        context=context,
        query=query
    )

    # Step 4: Ask the LLM
    target_llm = ChatOllama(model=model_name) if model_name else llm
    response = target_llm.invoke(message)

    print("\nAnswer:")
    print(response.content)

    return response.content


def retrival_chain_with_sources(
    query: str, 
    model_name: Optional[str] = None, 
    k: int = 3
) -> Dict[str, Any]:
    """Full retrieval chain returning both the answer and source citations for the UI."""
    target_model = model_name or MODEL
    r = get_retriever(k=k) or retriver
    
    if r is None:
        try:
            fallback_llm = ChatOllama(model=target_model)
            response = fallback_llm.invoke(query)
            return {
                "reply": response.content,
                "provider_used": f"Ollama Local ({target_model})",
                "model_used": target_model,
                "rag_sources": [],
                "context": ""
            }
        except Exception as e:
            return {
                "reply": f"⚠️ Could not generate response: {str(e)}",
                "provider_used": "RAG Error Handler",
                "model_used": target_model,
                "rag_sources": [],
                "context": ""
            }

    # Step 1: Retrieve relevant documents
    docs = r.invoke(query)
    print(f"[RAG] Retrieved {len(docs)} documents for query: '{query[:40]}...'")

    # Step 2: Format context and extract sources
    context = format_doc(docs)
    sources = extract_sources(docs)

    # Step 3: Create prompt
    message = prompt_template.format_messages(
        context=context,
        query=query
    )

    # Step 4: Ask the LLM
    try:
        target_llm = ChatOllama(model=target_model) if model_name else llm
        response = target_llm.invoke(message)
        reply_content = response.content

        return {
            "reply": reply_content,
            "provider_used": f"Pinecone RAG + Ollama ({target_model})",
            "model_used": target_model,
            "rag_sources": sources,
            "context": context
        }
    except Exception as e:
        print(f"[RAG] LLM generation error: {e}")
        return {
            "reply": f"⚠️ Pinecone retrieval succeeded ({len(sources)} documents), but Ollama generation encountered an error: {str(e)}",
            "provider_used": "RAG Error Handler",
            "model_used": target_model,
            "rag_sources": sources,
            "context": context
        }


if __name__ == "__main__":
    print("Retrieving ...")

    query = "What is SAR imagery, and how is it different from optical satellite imagery?"

    retrival_chain(query)