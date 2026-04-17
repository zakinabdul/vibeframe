from contextlib import asynccontextmanager
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agents import AgentEventBroker, VibeframeAgentPipeline
from app.config import settings
from app.groq_designer import GroqDesigner
from app.orchestrator import DesignOrchestrator
from app.paper_mcp import PaperMCPClient, PaperMCPConnectionError, PaperMCPProtocolError
from app.schemas import (
    CanvasArtboard,
    CanvasCurrentResponse,
    DesignRequest,
    DesignResponse,
    GenerateRequest,
    GenerateResponse,
    RefineRequest,
    RefineResponse,
    PaperOpenResponse,
    ResetSessionRequest,
)

paper_client = PaperMCPClient()
event_broker = AgentEventBroker()
agent_pipeline: VibeframeAgentPipeline | None = None


def get_agent_pipeline() -> VibeframeAgentPipeline:
    global agent_pipeline

    if agent_pipeline is not None:
        return agent_pipeline

    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    agent_pipeline = VibeframeAgentPipeline(
        paper_client=paper_client,
        event_broker=event_broker,
        groq_api_key=settings.groq_api_key,
        mistral_api_key=settings.mistral_api_key,
    )
    return agent_pipeline


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await paper_client.initialize()
        tools = await paper_client.list_tools()
        tool_names = [str(t.get("name", "")) for t in tools if t.get("name")]
        print(f"[startup] Paper MCP reachable. Discovered tools: {tool_names}")
    except Exception as exc:
        print(f"[startup] Paper MCP check failed: {exc}")
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.get("/health/dependencies")
async def health_dependencies() -> dict[str, str]:
    status = {
        "paper_mcp": "unknown",
        "groq": "configured" if bool(settings.groq_api_key) else "missing_api_key",
        "gemini_critic": "configured" if bool(settings.gemini_api_key) else "missing_api_key",
    }

    try:
        await paper_client.list_tools()
        status["paper_mcp"] = "reachable"
    except Exception as exc:
        status["paper_mcp"] = f"unreachable: {exc}"

    return status


@app.post("/api/design", response_model=DesignResponse)
async def design(payload: DesignRequest) -> DesignResponse:
    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    try:
        orchestrator = DesignOrchestrator(paper_client=paper_client, designer=GroqDesigner())
        return await orchestrator.run(payload.prompt)
    except PaperMCPConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PaperMCPProtocolError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Design orchestration failed: {exc}") from exc


@app.post("/generate", response_model=GenerateResponse)
async def generate(payload: GenerateRequest) -> GenerateResponse:
    try:
        result = await get_agent_pipeline().run_generate(
            payload.brief,
            source=payload.source,
            conversation_id=payload.conversation_id,
        )
        critique = result.get("critique", {}) if isinstance(result.get("critique"), dict) else {}
        return GenerateResponse(
            artboard_id=result.get("artboard_id"),
            palette_artboard_id=result.get("palette_artboard_id"),
            node_ids=result.get("node_ids", []),
            html_used=result.get("html_used", ""),
            round=int(result.get("round", 0)),
            critique=critique,
            done=bool(result.get("done", False)),
            assistant_message=str(result.get("assistant_message", "")),
            questions=[str(item) for item in result.get("questions", []) if isinstance(item, str)],
            conversation_stage=str(result.get("conversation_stage", "building")),
        )
    except PaperMCPConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PaperMCPProtocolError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generate pipeline failed: {exc}") from exc


@app.post("/refine", response_model=RefineResponse)
async def refine(payload: RefineRequest) -> RefineResponse:
    try:
        result = await get_agent_pipeline().run_refine(payload.artboard_id, payload.instruction)
        return RefineResponse(
            artboard_id=payload.artboard_id,
            node_ids=result.get("node_ids", []),
            html_used=result.get("html_used", ""),
            mode_used=result.get("mode_used"),
            tool_result=result.get("tool_result", {}),
        )
    except PaperMCPConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PaperMCPProtocolError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Refine failed: {exc}") from exc


@app.get("/canvas/current", response_model=CanvasCurrentResponse)
async def canvas_current() -> CanvasCurrentResponse:
    try:
        result = await get_agent_pipeline().get_current_canvas()
        return CanvasCurrentResponse(
            source=str(result.get("source", "unknown")),
            document_html=str(result.get("document_html", "")),
            artboards=[
                CanvasArtboard(id=str(item.get("id", "")), name=str(item.get("name", "Untitled")))
                for item in result.get("artboards", [])
                if isinstance(item, dict) and item.get("id")
            ],
            raw=result.get("raw") if isinstance(result.get("raw"), dict) else None,
        )
    except PaperMCPConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PaperMCPProtocolError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch current canvas: {exc}") from exc


@app.get("/stream")
async def stream() -> StreamingResponse:
    async def event_generator():
        queue = await event_broker.subscribe()
        try:
            yield "event: ready\ndata: {}\n\n"
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            event_broker.unsubscribe(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/paper/open", response_model=PaperOpenResponse)
async def paper_open() -> PaperOpenResponse:
    paper_path = settings.paper_desktop_path.strip()
    if not paper_path:
        return PaperOpenResponse(
            opened=False,
            message="PAPER_DESKTOP_PATH is not configured. Set it in backend/.env to enable one-click launch.",
        )

    if not os.path.exists(paper_path):
        return PaperOpenResponse(opened=False, message=f"Configured PAPER_DESKTOP_PATH does not exist: {paper_path}")

    try:
        os.startfile(paper_path)  # type: ignore[attr-defined]
        return PaperOpenResponse(opened=True, message="Paper Desktop launch command sent.")
    except Exception as exc:
        return PaperOpenResponse(opened=False, message=f"Failed to launch Paper Desktop: {exc}")


@app.post("/session/reset")
async def session_reset(payload: ResetSessionRequest) -> dict[str, str]:
    """Clear a conversation session so the next generate call creates a fresh canvas."""
    pipeline = get_agent_pipeline()
    conversation_id = payload.conversation_id or "default"
    if conversation_id in pipeline._conversation_sessions:
        del pipeline._conversation_sessions[conversation_id]
    return {"status": "reset", "conversation_id": conversation_id}
