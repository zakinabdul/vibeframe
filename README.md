# Vibeframe

Milestone 1 implementation: text prompt -> Groq Designer -> Paper MCP tool execution.

## Structure

- `backend` FastAPI orchestration service
- `frontend` Next.js control panel
- `.vscode/mcp.json` Paper MCP server config for local tooling

## 1) Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GROQ_API_KEY` in `backend/.env`.

Run backend:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 2) Frontend setup

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open `http://127.0.0.1:3000`.

## 3) Verify milestone 1

1. Ensure Paper Desktop MCP server is running at `http://127.0.0.1:29979/mcp`.
2. Start backend and frontend.
3. Submit a text prompt from frontend.
4. Confirm execution trace appears in UI.
5. Confirm elements are visibly created on Paper canvas.

## Deferred to later milestones

- Voice capture with Web Speech API
- Critic agent
- LangGraph multi-agent loops
