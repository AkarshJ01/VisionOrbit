import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import base64

from backend.models import ChatRequest, ChatResponse, ImageAnalysisRequest
from backend.vision_service import vision_service

app = FastAPI(
    title="VisionOrbit API",
    description="State-of-the-art Multimodal Vision & Prompting Interface",
    version="1.0.0"
)

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "images"), exist_ok=True)

@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "VisionOrbit API is running. Frontend index.html not found."}

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "VisionOrbit Multimodal AI"}

@app.get("/api/models")
async def get_models(ollama_url: Optional[str] = None):
    """Retrieve available models and provider capabilities."""
    ollama_info = await vision_service.check_ollama_status(ollama_url)
    has_env_openai = bool(os.getenv("OPENAI_API_KEY", "").strip())
    has_env_tavily = bool(os.getenv("TAVILY_API_KEY", "").strip())
    
    return {
        "providers": [
            {
                "id": "builtin",
                "name": "VisionOrbit Smart Engine",
                "badge": "Active & Fast",
                "available": True,
                "models": [
                    {"id": "visionorbit-core", "name": "VisionOrbit Multimodal Core", "description": "High-fidelity instant visual analyzer & prompt engine"}
                ]
            },
            {
                "id": "openai",
                "name": "OpenAI Vision (GPT-4o)",
                "badge": "Configured in .env" if has_env_openai else "API Key Required",
                "available": True,
                "configured": has_env_openai,
                "models": [
                    {"id": "gpt-4o", "name": "GPT-4o (Omni Vision)", "description": "High-intelligence flagship vision model"},
                    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "description": "Fast & lightweight multimodal vision model"},
                    {"id": "gpt-4-turbo", "name": "GPT-4 Turbo with Vision", "description": "High-accuracy vision reasoning"}
                ]
            },
            {
                "id": "ollama",
                "name": "Ollama Local Vision",
                "badge": "Connected" if ollama_info.get("available") else "Offline",
                "available": ollama_info.get("available", False),
                "url": ollama_info.get("url"),
                "models": [
                    {"id": m, "name": f"Ollama {m}", "description": "Local on-device multimodal model"} 
                    for m in ollama_info.get("vision_models", ["llava", "llama3.2-vision", "moondream"])
                ]
            }
        ],
        "features": {
            "has_tavily": has_env_tavily,
            "has_openai": has_env_openai,
            "has_ollama": ollama_info.get("available", False)
        }
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main multimodal chat and image analysis endpoint."""
    try:
        result = await vision_service.generate_response(
            prompt=request.prompt,
            images=request.images or [],
            history=request.history or [],
            provider=request.provider or "auto",
            model=request.model or "gpt-4o",
            api_key=request.apiKey,
            ollama_base_url=request.ollamaBaseUrl,
            tavily_key=request.tavilyApiKey,
            use_web_search=request.useWebSearch or False,
            system_prompt=request.systemPrompt
        )
        return ChatResponse(
            reply=result.get("reply", ""),
            provider_used=result.get("provider_used", "VisionOrbit Engine"),
            model_used=result.get("model_used", "VisionOrbit Core"),
            image_metadata=result.get("image_metadata", []),
            search_sources=result.get("search_sources", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file and return its base64 data URL with metadata."""
    try:
        contents = await file.read()
        mime_type = file.content_type or "image/png"
        b64 = base64.b64encode(contents).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"
        metadata = vision_service.parse_image_info(data_url)
        return {
            "filename": file.filename,
            "data_url": data_url,
            "mime_type": mime_type,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

# Mount static files at /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def main():
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting VisionOrbit server at http://localhost:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    main()
