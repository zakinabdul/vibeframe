# Vibeframe

Vibeframe is a next-generation AI-powered web design orchestrator and conversational agent. By leveraging multiple advanced agent pipelines and a real-time voice/text conversational interface, Vibeframe acts as an autonomous design partner that interviews users, captures their vision, and translates it into functional UI designs.

## High-Level Architecture

Vibeframe is composed of a tightly integrated modern technology stack:

- **Frontend (`frontend/`)**: 
  - Built with **Next.js** and **React**.
  - Features a sophisticated hybrid text and voice conversational interface.
  - Supports continuous "Hold-to-Talk" real-time voice streaming using the Web Speech API and WebSocket chunking for extremely low-latency interactions.

- **Backend (`backend/`)**: 
  - Fast, async server built with **FastAPI**.
  - **LangGraph** orchestrates the core agent pipeline (`agentv2.py`), processing conversational turns across multiple stages (Intake, Palette Selection, Content Gathering, and Design Generation).
  - Integrates with **Groq STT (Whisper)** for hyper-fast, high-accuracy conversational fallback transcription.
  - **Paper MCP Client** to seamlessly launch and interact with the local Paper application canvas.

## Key Features

- 🎙️ **Real-time Voice Conversational UI:** Speak your ideas directly to the system using lightning-fast WebSocket-based audio chunking and sub-second silence capture capabilities.
- 🎨 **Generative Design Pipeline:** Fully autonomous LangGraph agent pipeline capable of teasing out project requirements and orchestrating artboards and layout nodes autonomously.
- ⚡ **Lightning Fast Responses:** State-of-the-art TTS/STT pipelines paired with `llama` and `mistral` model routing via Groq ensure zero perceptible delay.
- 🔌 **Deep "Paper" Integration:** Hooks directly into a native design client (`paper://`) to instantly view and edit the AI-generated interface on desktop.

## Getting Started

### Prerequisites
1. Node.js `^18.0.0`
2. Python `^3.10`
3. A `.env` file populated with your API credentials (Groq, Langsmith, etc.) inside the `backend` directory.

### Quickstart

#### 1. Start the Backend
Navigate to the backend directory, install the dependencies, and start the FastAPI uvicorn daemon:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # (or .\.venv\Scripts\activate on Windows)
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Start the Frontend
In a new terminal window, initialize and run the Next.js frontend:

```bash
cd frontend
npm install
npm run dev
```

The application will be running locally at `http://localhost:3000`.

## Workflow Overview

1. **Intake Phase**: Upon connection, Vibeframe asks probing questions to ascertain the vibe and goals.
2. **Palette Generation**: Based on the context gathered, multi-theme design tokens are generated.
3. **Content Gathering**: Specific copy points (hero text, features, target audience) are acquired.
4. **Build & Refine**: The agent routes final prompts out to the design node and updates the UI canvas directly over the Paper MCP protocol.

## Contributing

Make sure all new agent behaviors follow the structural checklists outlined in `agentv2.py` (e.g., strong state typing, isolated nodes, and normalized JSON schema definitions).
