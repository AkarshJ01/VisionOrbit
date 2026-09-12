# 🪐 VisionOrbit

**VisionOrbit** is a modern, multimodal AI visual intelligence web application inspired by OpenAI's clean conversational interface. It enables users to ask questions, prompt the AI, upload images (via drag-and-drop, file picker, or clipboard paste `⌘V`), and receive in-depth visual analyses, OCR text extraction, UI/UX breakdowns, and telemetry insights.

---

## 🌟 Key Features

1. **OpenAI-Style Prompting Handle & Interface**:
   - Clean obsidian dark theme with smooth glassmorphism and subtle neon glow accents.
   - Auto-expanding multiline prompt input dock with keyboard shortcuts (`Enter` to submit, `Shift+Enter` for new lines, `⌘K` for New Chat).
   - Real-time animated streaming response cards with markdown formatting, syntax-highlighted code blocks, telemetry tables, and one-click copy/speak actions.
   - Welcoming Hero interface with quick starter suggestion cards for deep visual inspection, OCR transcription, UI design review, and chart/graph analytics.

2. **Multimodal Image Analysis & Staging**:
   - **Multiple Upload Channels**:
     - Drag & drop images anywhere on the window or into the hero dropzone.
     - Direct clipboard paste (`Cmd+V` / `Ctrl+V`) for instant screenshot analysis.
     - Attachment button (`📎`) supporting `.png`, `.jpg`, `.jpeg`, `.webp`, and `.gif`.
   - Staging tray showing image thumbnail cards before sending.
   - Click-to-zoom interactive **Lightbox Modal** for full-resolution inspection.

3. **Multi-Provider Neural Architecture**:
   - **OpenAI Multimodal Vision** (`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`): High-intelligence live vision reasoning.
   - **Ollama Local Vision** (`llava`, `llama3.2-vision`, `moondream`, etc.): On-device private multimodal intelligence.
   - **VisionOrbit Intelligence Engine**: Fast built-in vision telemetry and heuristic analysis engine that works out-of-the-box with zero API key dependencies.
   - **Tavily Web Search Enrichment**: Real-time web fact-checking for multimodal queries.

4. **Chat History & Management**:
   - Persistent conversation threads saved in browser `localStorage`.
   - Export chats as Markdown (`.md`).
   - Theme switch between Obsidian Dark and Clean Slate Light.

---

## 🚀 Quickstart Guide

### 1. Installation

Ensure Python 3.13+ and `uv` are installed:

```bash
# Install dependencies
uv sync
```

### 2. Start the VisionOrbit Application

Run the server:

```bash
uv run python main.py
```

The application will start immediately at:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## ⚙️ Configuration (Optional)

You can configure API keys either via a `.env` file or directly inside the web UI via **Settings (⚙️)**:

```ini
# .env (Optional)
PORT=8000
OPENAI_API_KEY=sk-proj-...
OLLAMA_BASE_URL=http://localhost:11434
TAVILY_API_KEY=tvly-...
```

---

## 🧪 Testing Backend API

To verify backend multimodal processing and telemetry extraction:

```bash
uv run python test_app.py
```
