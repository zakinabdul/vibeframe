from contextlib import asynccontextmanager
import asyncio
import io
import json
import logging
import os
import time
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import Groq
from pydantic import SecretStr

from agents import (
    AgentEventBroker as AgentEventBrokerV1,
    AgentEventBrokerV2,
    VibeframeAgentPipeline as VibeframeAgentPipelineV1,
    VibeframeAgentPipelineV2,
)
from app.config import settings
from app.groq_designer import GroqDesigner
from app.orchestrator import DesignOrchestrator
from app.paper_mcp import PaperMCPClient, PaperMCPConnectionError, PaperMCPProtocolError
from app.schemas import (
    CanvasArtboard,
    CanvasCurrentResponse,
    Critique,
    DesignRequest,
    DesignResponse,
    GenerateRequest,
    GenerateResponse,
    RefineRequest,
    RefineResponse,
    PaperOpenResponse,
    ResetSessionRequest,
    TranscribeResponse,
)

paper_client = PaperMCPClient()
event_broker = AgentEventBrokerV2() if AgentEventBrokerV2 is not None else AgentEventBrokerV1()
legacy_event_broker = AgentEventBrokerV1()
agent_pipeline: VibeframeAgentPipelineV2 | VibeframeAgentPipelineV1 | None = None
legacy_agent_pipeline: VibeframeAgentPipelineV1 | None = None
stt_client: Groq | None = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
)
logger = logging.getLogger(__name__)


# Design-vocabulary prompt to improve transcription accuracy for domain terms
STT_CONTEXT_PROMPT = (
    "Vibeframe, landing page, SaaS, portfolio, ecommerce, palette, artboard, "
    "hero section, navigation, typography, gradient, dark mode, glassmorphism, "
    "call to action, CTA, responsive, wireframe, mockup, UI, UX"
)


def get_stt_client() -> Groq:
    global stt_client

    if stt_client is not None:
        return stt_client

    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    stt_client = Groq(api_key=settings.groq_api_key)
    return stt_client


def get_agent_pipeline() -> VibeframeAgentPipelineV2 | VibeframeAgentPipelineV1:
    global agent_pipeline

    if agent_pipeline is not None:
        return agent_pipeline

    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    broker: Any = event_broker
    if VibeframeAgentPipelineV2 is not None and AgentEventBrokerV2 is not None:
        agent_pipeline = VibeframeAgentPipelineV2(
            groq_api_key=SecretStr(settings.groq_api_key),
            paper_client=paper_client,
            event_broker=broker,
        )
    else:
        agent_pipeline = VibeframeAgentPipelineV1(
            paper_client=paper_client,
            event_broker=broker,
            groq_api_key=settings.groq_api_key,
        )
    return agent_pipeline


def get_legacy_agent_pipeline() -> VibeframeAgentPipelineV1:
    global legacy_agent_pipeline

    if legacy_agent_pipeline is not None:
        return legacy_agent_pipeline

    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    legacy_agent_pipeline = VibeframeAgentPipelineV1(
        paper_client=paper_client,
        event_broker=legacy_event_broker,
        groq_api_key=settings.groq_api_key,
    )
    return legacy_agent_pipeline


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm STT client on startup to avoid first-request latency
    try:
        if settings.groq_api_key:
            get_stt_client()
            logger.info("[main.py] startup STT client pre-warmed (%s)", settings.groq_transcription_model)
    except Exception as exc:
        logger.warning("[main.py] startup STT client warm failed: %s", exc)

    try:
        await paper_client.initialize()
        tools = await paper_client.list_tools()
        tool_names = [str(t.get("name", "")) for t in tools if t.get("name")]
        logger.info("[main.py] startup Paper MCP reachable. tools=%s", tool_names)
    except Exception as exc:
        logger.warning("[main.py] startup Paper MCP check failed: %s", exc)
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
    logger.info("[main.py] GET /health")
    return {"status": "ok", "environment": settings.app_env}


@app.get("/health/dependencies")
async def health_dependencies() -> dict[str, str]:
    logger.info("[main.py] GET /health/dependencies")
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
    logger.info("[main.py] POST /api/design")
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
    logger.info("[main.py] POST /generate source=%s conversation_id=%s", payload.source, payload.conversation_id or "default")
    try:
        result = await get_agent_pipeline().run_generate(
            payload.brief,
            source=payload.source,
            conversation_id=payload.conversation_id,
        )
        critique_raw = result.get("critique", {}) if isinstance(result.get("critique"), dict) else {}
        critique = Critique(
            score=int(critique_raw.get("score", 0) or 0),
            issues=[str(x) for x in critique_raw.get("issues", []) if str(x).strip()] if isinstance(critique_raw.get("issues", []), list) else [],
            suggestions=[str(x) for x in critique_raw.get("suggestions", []) if str(x).strip()] if isinstance(critique_raw.get("suggestions", []), list) else [],
        )
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


@app.post("/v2/generate", response_model=GenerateResponse)
async def generate_v2(payload: GenerateRequest) -> GenerateResponse:
    logger.info("[main.py] POST /v2/generate")
    return await generate(payload)


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate_v1(payload: GenerateRequest) -> GenerateResponse:
    logger.info("[main.py] POST /v1/generate source=%s conversation_id=%s", payload.source, payload.conversation_id or "default")
    try:
        result = await get_legacy_agent_pipeline().run_generate(
            payload.brief,
            source=payload.source,
            conversation_id=payload.conversation_id,
        )
        critique_raw = result.get("critique", {}) if isinstance(result.get("critique"), dict) else {}
        critique = Critique(
            score=int(critique_raw.get("score", 0) or 0),
            issues=[str(x) for x in critique_raw.get("issues", []) if str(x).strip()] if isinstance(critique_raw.get("issues", []), list) else [],
            suggestions=[str(x) for x in critique_raw.get("suggestions", []) if str(x).strip()] if isinstance(critique_raw.get("suggestions", []), list) else [],
        )
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
        raise HTTPException(status_code=500, detail=f"Generate v1 pipeline failed: {exc}") from exc


@app.post("/refine", response_model=RefineResponse)
async def refine(payload: RefineRequest) -> RefineResponse:
    logger.info("[main.py] POST /refine artboard_id=%s", payload.artboard_id)
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


@app.post("/v2/refine", response_model=RefineResponse)
async def refine_v2(payload: RefineRequest) -> RefineResponse:
    logger.info("[main.py] POST /v2/refine artboard_id=%s", payload.artboard_id)
    return await refine(payload)


@app.post("/v1/refine", response_model=RefineResponse)
async def refine_v1(payload: RefineRequest) -> RefineResponse:
    logger.info("[main.py] POST /v1/refine artboard_id=%s", payload.artboard_id)
    try:
        result = await get_legacy_agent_pipeline().run_refine(payload.artboard_id, payload.instruction)
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
        raise HTTPException(status_code=500, detail=f"Refine v1 failed: {exc}") from exc


@app.get("/canvas/current", response_model=CanvasCurrentResponse)
async def canvas_current() -> CanvasCurrentResponse:
    logger.info("[main.py] GET /canvas/current")
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
    logger.info("[main.py] GET /stream")
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
    logger.info("[main.py] POST /paper/open")
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
    logger.info("[main.py] POST /session/reset conversation_id=%s", payload.conversation_id or "default")
    pipeline = get_agent_pipeline()
    conversation_id = payload.conversation_id or "default"
    if conversation_id in pipeline._conversation_sessions:
        del pipeline._conversation_sessions[conversation_id]
    voice_sessions = getattr(pipeline, "_voice_sessions", None)
    if isinstance(voice_sessions, dict) and conversation_id in voice_sessions:
        del voice_sessions[conversation_id]
    return {"status": "reset", "conversation_id": conversation_id}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
) -> TranscribeResponse:
    t0 = time.perf_counter()
    logger.info("[main.py] POST /transcribe filename=%s content_type=%s", audio.filename, audio.content_type)

    if not settings.groq_api_key:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio payload received.")

    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio payload is too large (max 25MB).")

    file_name = audio.filename or "voice.webm"
    content_type = audio.content_type or "audio/webm"
    transcription_model = settings.groq_transcription_model

    def _transcribe_sync() -> str:
        client = get_stt_client()
        file_buffer = io.BytesIO(audio_bytes)
        file_buffer.name = file_name

        # Use design-vocabulary context prompt for better accuracy
        effective_prompt = prompt.strip() if prompt and prompt.strip() else STT_CONTEXT_PROMPT

        kwargs: dict[str, Any] = {
            "model": transcription_model,
            "file": (file_name, audio_bytes, content_type),
            "temperature": 0,
            "response_format": "text",
            "prompt": effective_prompt,
        }
        if language and language.strip():
            kwargs["language"] = language.strip()

        try:
            response = client.audio.transcriptions.create(**kwargs)
        except TypeError:
            # Some SDK versions accept file-like objects instead of tuple payloads.
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["file"] = file_buffer
            response = client.audio.transcriptions.create(**fallback_kwargs)

        # response_format="text" returns a plain string in some SDK versions
        if isinstance(response, str):
            return response.strip()
        text = getattr(response, "text", "")
        return str(text or "").strip()

    try:
        transcript_text = await asyncio.to_thread(_transcribe_sync)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {exc}") from exc

    elapsed = time.perf_counter() - t0
    logger.info("[main.py] POST /transcribe completed in %.2fs chars=%d", elapsed, len(transcript_text))

    return TranscribeResponse(
        text=transcript_text,
        provider="groq",
        model=transcription_model,
    )


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time chunked audio transcription.

    Protocol:
    - Client sends binary audio chunk frames as bytes
    - Client sends JSON text frame {"action": "finalize"} to trigger transcription
    - Client sends JSON text frame {"action": "reset"} to clear accumulated audio
    - Server responds with JSON {"text": "...", "final": true/false, "elapsed": float}
    """
    await websocket.accept()
    logger.info("[main.py] WS /ws/transcribe connected")

    if not settings.groq_api_key:
        await websocket.send_json({"error": "Missing GROQ_API_KEY"})
        await websocket.close(code=1008)
        return

    audio_chunks: list[bytes] = []
    transcription_model = settings.groq_transcription_model

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # Binary frame → audio chunk
            if "bytes" in message and message["bytes"]:
                audio_chunks.append(message["bytes"])
                continue

            # Text frame → control command
            if "text" in message and message["text"]:
                try:
                    cmd = json.loads(message["text"])
                except (json.JSONDecodeError, TypeError):
                    continue

                action = cmd.get("action", "")

                if action == "reset":
                    audio_chunks.clear()
                    await websocket.send_json({"text": "", "final": False, "elapsed": 0})
                    continue

                if action == "finalize":
                    if not audio_chunks:
                        await websocket.send_json({"text": "", "final": True, "elapsed": 0})
                        continue

                    t0 = time.perf_counter()
                    combined = b"".join(audio_chunks)
                    audio_chunks.clear()

                    if len(combined) < 256:
                        await websocket.send_json({"text": "", "final": True, "elapsed": 0})
                        continue

                    language = cmd.get("language", "")

                    def _ws_transcribe_sync() -> str:
                        client = get_stt_client()
                        kwargs: dict[str, Any] = {
                            "model": transcription_model,
                            "file": ("voice.webm", combined, "audio/webm"),
                            "temperature": 0,
                            "response_format": "text",
                            "prompt": STT_CONTEXT_PROMPT,
                        }
                        if language:
                            kwargs["language"] = language

                        try:
                            resp = client.audio.transcriptions.create(**kwargs)
                        except TypeError:
                            buf = io.BytesIO(combined)
                            buf.name = "voice.webm"
                            fallback = dict(kwargs)
                            fallback["file"] = buf
                            resp = client.audio.transcriptions.create(**fallback)

                        if isinstance(resp, str):
                            return resp.strip()
                        return str(getattr(resp, "text", "") or "").strip()

                    try:
                        text = await asyncio.to_thread(_ws_transcribe_sync)
                        elapsed = time.perf_counter() - t0
                        logger.info("[main.py] WS transcribe chunk %.2fs chars=%d", elapsed, len(text))
                        await websocket.send_json({"text": text, "final": True, "elapsed": round(elapsed, 3)})
                    except Exception as exc:
                        logger.error("[main.py] WS transcribe error: %s", exc)
                        await websocket.send_json({"error": str(exc), "final": True, "elapsed": 0})

    except WebSocketDisconnect:
        logger.info("[main.py] WS /ws/transcribe disconnected")
    except Exception as exc:
        logger.error("[main.py] WS /ws/transcribe unexpected error: %s", exc)
