import os
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

import rag

load_dotenv()

class RagService:
    def __init__(self):
        self.index_name = os.getenv("INDEX_NAME", "")
        self.pinecone_key = os.getenv("PINECONE_API_KEY", "")
        self.default_model = "gpt-oss:20b"

    def retrieve_documents(self, query: str, k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        """Retrieve relevant document chunks and formatted sources from Pinecone via rag.py."""
        return rag.retrieve_documents(query, k=k)

    def retrival_chain(self, query: str, model_name: Optional[str] = None) -> str:
        """Directly invoke retrival_chain from rag.py."""
        return rag.retrival_chain(query, model_name=model_name)

    async def execute_rag_chain(
        self,
        query: str,
        model_name: Optional[str] = None,
        k: int = 3,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the full retrieval-augmented generation chain asynchronously using rag.py."""
        target_model = model_name or self.default_model
        
        # Run sync LangChain chain in threadpool to keep FastAPI async loop responsive
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: rag.retrival_chain_with_sources(query, model_name=target_model, k=k)
        )
        return result

rag_service = RagService()
