import asyncio
import rag
from backend.vision_service import vision_service
from backend.rag_service import rag_service

async def test_rag():
    print("1. Testing rag.py direct retrieval...")
    context, sources = rag.retrieve_documents("What is SAR imagery?", k=2)
    print(f"Retrieved {len(sources)} sources from Pinecone index '{os.getenv('INDEX_NAME')}'")
    for s in sources:
        print(f" - [{s['filename']}] Page {s['page']}: {s['snippet'][:80]}...")
    assert len(sources) > 0, "Expected at least 1 retrieved source from Pinecone"

    print("\n2. Testing rag.py retrival_chain_with_sources...")
    rag_out = rag.retrival_chain_with_sources(
        query="What is SAR imagery, and how is it different from optical satellite imagery?",
        model_name="gpt-oss:20b",
        k=2
    )
    print("Provider used:", rag_out.get("provider_used"))
    print("Model used:", rag_out.get("model_used"))
    print(f"RAG sources attached: {len(rag_out.get('rag_sources', []))}")
    print("Answer excerpt:\n", rag_out.get("reply", "")[:200], "...\n")
    assert len(rag_out.get("rag_sources", [])) > 0
    assert len(rag_out.get("reply", "")) > 50

    print("\n3. Testing VisionService generate_response with RAG enabled...")
    res = await vision_service.generate_response(
        prompt="What are the key applications of SAR for forest biomass monitoring?",
        images=[],
        provider="ollama",
        model="gpt-oss:20b",
        use_rag=True
    )
    print("Provider used:", res.get("provider_used"))
    print(f"RAG sources attached: {len(res.get('rag_sources', []))}")
    print("Answer excerpt:\n", res.get("reply", "")[:200], "...\n")
    assert len(res.get("rag_sources", [])) > 0

    print("✅ All Pinecone RAG & rag.py integration tests passed successfully!")

if __name__ == "__main__":
    import os
    asyncio.run(test_rag())
