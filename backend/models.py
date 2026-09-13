from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
import time

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str = Field(..., description="Textual message content")
    images: Optional[List[str]] = Field(default=[], description="List of base64 data URLs or image URLs")
    detection_results: Optional[Dict[str, Any]] = Field(default=None, description="Prior object detection findings")
    annotated_image: Optional[str] = Field(default=None, description="Prior annotated image")
    timestamp: Optional[float] = Field(default_factory=time.time)

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt or question")
    images: Optional[List[str]] = Field(default=[], description="List of base64 images or image URLs")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Prior conversation context")
    provider: Optional[str] = Field(default="auto", description="'openai', 'ollama', 'builtin', or 'auto'")
    model: Optional[str] = Field(default="gpt-oss:20b", description="Target model name")
    apiKey: Optional[str] = Field(default=None, description="Client-provided OpenAI API Key (optional)")
    ollamaBaseUrl: Optional[str] = Field(default=None, description="Client-provided Ollama Base URL (optional)")
    tavilyApiKey: Optional[str] = Field(default=None, description="Client-provided Tavily API Key (optional)")
    useWebSearch: Optional[bool] = Field(default=False, description="Whether to enrich with Tavily web search")
    useRag: Optional[bool] = Field(default=True, description="Whether to retrieve context from Pinecone Knowledge Base")
    systemPrompt: Optional[str] = Field(default=None, description="Custom system instruction")
    confThreshold: Optional[float] = Field(default=0.20, description="YOLO object detection confidence threshold")
    useDetection: Optional[bool] = Field(default=True, description="Whether to execute real-time YOLO-OBB object detection")

class RagRequest(BaseModel):
    query: str = Field(..., description="Search query or question for Pinecone RAG")
    model: Optional[str] = Field(default="gpt-oss:20b", description="Target Ollama model name")
    k: Optional[int] = Field(default=3, description="Number of document chunks to retrieve")

class ImageAnalysisRequest(BaseModel):
    image: str = Field(..., description="Base64 image data URL")
    aspects: Optional[List[str]] = Field(default=["description", "objects", "text", "colors", "recommendations"])
    provider: Optional[str] = Field(default="auto")
    apiKey: Optional[str] = None
    confThreshold: Optional[float] = 0.20

class ChatResponse(BaseModel):
    reply: str
    provider_used: str
    model_used: str
    annotated_image: Optional[str] = None
    detection_results: Optional[Dict[str, Any]] = None
    image_metadata: Optional[List[Dict[str, Any]]] = []
    search_sources: Optional[List[Dict[str, Any]]] = []
    rag_sources: Optional[List[Dict[str, Any]]] = []
    timestamp: float = Field(default_factory=time.time)

