import asyncio
import base64
from backend.vision_service import vision_service
from backend.models import ChatRequest

async def test_backend():
    print("Testing VisionService...")
    
    # 1. Test Text Only
    res_text = await vision_service.generate_response(
        prompt="Hello, what can you do?",
        images=[],
        provider="builtin"
    )
    print("Text only response provider:", res_text.get("provider_used"))
    assert "VisionOrbit" in res_text.get("reply", "")

    # 2. Test with sample 1x1 PNG base64 image
    # 1x1 transparent PNG:
    sample_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    res_image = await vision_service.generate_response(
        prompt="Extract text and inspect the UI layout from this screenshot",
        images=[sample_png],
        provider="builtin"
    )
    print("Image analysis response provider:", res_image.get("provider_used"))
    print("Image metadata:", res_image.get("image_metadata"))
    assert len(res_image.get("image_metadata", [])) == 1
    assert "Multimodal Visual Analysis" in res_image.get("reply", "")

    print("\n✅ All VisionService backend tests passed successfully!")

if __name__ == "__main__":
    asyncio.run(test_backend())
