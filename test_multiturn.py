import os
import asyncio
from backend.vision_service import vision_service
from backend.models import ChatMessage

async def run_multiturn_test():
    print("🚀 Starting Multi-Turn Vision Inquiry Verification...\n")

    test_image_path = os.path.expanduser("~/Desktop/1011.tif")
    if not os.path.exists(test_image_path):
        test_image_path = os.path.abspath("data/raw/images/1.tif")
    if not os.path.exists(test_image_path):
        print(f"⚠️ Test image not found at {test_image_path}")
        return

    # --- Turn 1: Initial Image Upload & Detection ---
    print("--- [TURN 1: Uploading GeoTIFF & Analyzing Targets] ---")
    turn1_res = await vision_service.generate_response(
        prompt="Detect all vehicles and objects in this satellite image.",
        images=[test_image_path],
        history=[],
        provider="builtin",
        use_detection=True,
        use_rag=False
    )
    
    turn1_reply = turn1_res.get("reply", "")
    turn1_det = turn1_res.get("detection_results", {})
    total_dets = turn1_det.get("total_detections", 0)
    print(f"Turn 1 Provider: {turn1_res.get('provider_used')}")
    print(f"Turn 1 Total Detections: {total_dets}")
    print(f"Turn 1 Class Breakdown: {turn1_det.get('class_counts')}")
    assert total_dets > 0, "Expected positive detection count in Turn 1"

    # Build conversation history
    history = [
        ChatMessage(
            role="user",
            content="Detect all vehicles and objects in this satellite image.",
            images=[test_image_path]
        ),
        ChatMessage(
            role="assistant",
            content=turn1_reply,
            detection_results=turn1_det,
            annotated_image=turn1_res.get("annotated_image")
        )
    ]

    # --- Turn 2: Follow-up Question about Vans (NO IMAGE ATTACHED) ---
    print("\n--- [TURN 2: Follow-up Query: 'Where are the vans located?'] ---")
    turn2_res = await vision_service.generate_response(
        prompt="Where are the vans located in the image and what are their coordinates?",
        images=[], # No image uploaded!
        history=history,
        provider="builtin",
        use_detection=True,
        use_rag=False
    )

    turn2_reply = turn2_res.get("reply", "")
    print(f"Turn 2 Provider: {turn2_res.get('provider_used')}")
    print("Turn 2 Reply Snippet:\n", turn2_reply[:500])
    
    # Verify that Turn 2 accurately identified vans from Turn 1's detection context
    assert "van" in turn2_reply.lower(), "Turn 2 should reference the detected vans"
    assert "Target #" in turn2_reply or "px" in turn2_reply or "coordinate" in turn2_reply.lower(), "Turn 2 should include coordinates/metrics"

    # Append Turn 2 to history
    history.append(ChatMessage(role="user", content="Where are the vans located in the image and what are their coordinates?"))
    history.append(ChatMessage(role="assistant", content=turn2_reply))

    # --- Turn 3: Follow-up Question about Counts (NO IMAGE ATTACHED) ---
    print("\n--- [TURN 3: Follow-up Query: 'How many small vehicles vs large vehicles?'] ---")
    turn3_res = await vision_service.generate_response(
        prompt="How many small vehicles are there compared to large vehicles?",
        images=[], # No image uploaded!
        history=history,
        provider="builtin",
        use_detection=True,
        use_rag=False
    )

    turn3_reply = turn3_res.get("reply", "")
    print(f"Turn 3 Provider: {turn3_res.get('provider_used')}")
    print("Turn 3 Reply Snippet:\n", turn3_reply[:500])
    assert "small-vehicle" in turn3_reply or "small vehicle" in turn3_reply.lower(), "Turn 3 should reference small vehicles"
    assert "large-vehicle" in turn3_reply or "large vehicle" in turn3_reply.lower(), "Turn 3 should reference large vehicles"

    # --- Turn 4: Test with Ollama (if available) ---
    ollama_status = await vision_service.check_ollama_status()
    if ollama_status.get("available"):
        print("\n--- [TURN 4: Ollama Multi-Turn Test] ---")
        turn4_res = await vision_service.generate_response(
            prompt="Based on the detected vehicles in our satellite image, summarize their distribution.",
            images=[],
            history=history,
            provider="ollama",
            model="gpt-oss:20b",
            use_detection=True,
            use_rag=False
        )
        print("Turn 4 Ollama Reply Snippet:\n", turn4_res.get("reply", "")[:400])
        assert len(turn4_res.get("reply", "")) > 10, "Ollama should return non-empty multi-turn response"

    print("\n🎉 Multi-Turn Image Memory Verification Completed Successfully!")

if __name__ == "__main__":
    asyncio.run(run_multiturn_test())
