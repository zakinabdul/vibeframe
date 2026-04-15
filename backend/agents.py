import asyncio
import json
from typing import Any, Literal, TypedDict

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.paper_mcp import PaperMCPClient, PaperMCPProtocolError


DESIGNER_SYSTEM_PROMPT = """You are Vibeframe's Senior UI Designer Agent.
Your output is used to render production-quality UI directly on Paper Design canvas through MCP tools.

Design mission:
- Create stunning, visually rich landing pages, not minimal wireframes.
- Think like a senior product designer focused on polish, hierarchy, rhythm, and conversion.

Required page structure (all sections must exist):
1. Navigation bar
2. Hero section
3. Features section with exactly 3 cards
4. CTA section
5. Footer

Mandatory design system:
- Background: #0a0a0a
- Primary accent (indigo): #6366f1
- Card background: #1a1a1a
- Main text: #ffffff
- Muted text: #888888
- Borders: #2a2a2a

Typography and hierarchy:
- Hero headline: 64px-72px, bold
- Section headings: 36px
- Body text: 16px-18px
- Keep strong contrast and spacing rhythm

Visual depth requirements:
- Use hero text gradients (inline styles)
- Add card borders and depth
- Add button hover affordance via inline transition states where possible
- Add subtle background patterns/overlays using inline CSS
- Use generous spacing and padding; never cramped layouts

Layout rules:
- Use flexbox for major layout groups
- Nav: horizontal + space-between
- Hero: centered column composition
- Features: horizontal card row with visible gap
- Ensure padding and section separation are substantial

CRITICAL PAPER MCP RULES:
1. Always use inline styles on every HTML element inside write_html
2. The HTML for write_html must be ONE complete wrapping div containing all sections
3. Never use CSS selectors in update_styles
4. update_styles only accepts real node IDs from previous MCP responses
5. Never invent selector-like IDs such as "$LAST_ARTBOARD_ID > h1", ".cta", "#hero"
6. For existing design edits, preserve structure and improve quality in-place via updated full HTML

Output contract:
- Return strict JSON only: {"summary": string, "html": string}
- html must be a single complete root <div> with all required sections and inline styles
- No markdown fences, no extra keys
"""


CRITIC_SYSTEM_PROMPT = """You are Vibeframe Critic Agent.
Evaluate current landing-page quality and return strict JSON only:
{
  "score": number,
  "issues": string[],
  "suggestions": string[]
}

Scoring rubric:
- 1-3: broken, missing sections, poor hierarchy
- 4-6: functional but weak aesthetics and layout
- 7-8: strong baseline with minor polish gaps
- 9-10: excellent, production-quality visual system and hierarchy

Focus checks:
- All required sections exist (Nav, Hero, Features x3 cards, CTA, Footer)
- Visual hierarchy is strong and readable
- Color system follows spec
- Spacing is generous and uncluttered
- Buttons and cards feel intentionally styled
- Overall page looks premium and coherent

Be concise and actionable.
"""


class AgentEventBroker:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    async def publish(self, event: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue


class PipelineState(TypedDict, total=False):
    status: Literal["initialized", "designing", "critiquing", "refining", "complete", "degraded"]
    source: Literal["text", "voice"]
    brief: str
    artboard_id: str | None
    node_ids: list[str]
    round: int
    retry_count: int
    max_rounds: int
    max_critic_retries: int
    target_score: int
    next_action: Literal["refine", "end", "retry_critic"]
    critique: dict[str, Any]
    done: bool
    html_used: str
    current_html: str
    create_new: bool


class ConversationSession(TypedDict, total=False):
    stage: Literal["intake", "palette_requested", "building", "done"]
    palette_artboard_id: str
    last_user_brief: str
    palette_summary: str
    questions: list[str]


class VibeframeAgentPipeline:
    def __init__(self, paper_client: PaperMCPClient, event_broker: AgentEventBroker, groq_api_key: str) -> None:
        self.paper_client = paper_client
        self.event_broker = event_broker
        self.gemini_api_key = settings.gemini_api_key
        self.gemini_critic_model = settings.gemini_critic_model
        self.gemini_api_base = settings.gemini_api_base.rstrip("/")
        self.designer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.35, api_key=groq_api_key)
        self.critic_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=groq_api_key)
        self._conversation_sessions: dict[str, ConversationSession] = {}
        self.graph = self._build_graph()

    async def run_generate(
        self,
        brief: str,
        source: Literal["text", "voice"] = "text",
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        session_id = conversation_id or "default"
        session = self._conversation_sessions.setdefault(session_id, {"stage": "intake", "questions": []})

        voice_result = await self._handle_voice_turn(session_id=session_id, brief=brief, source=source, session=session)
        if voice_result is not None:
            return voice_result

        initial_state: PipelineState = {
            "status": "initialized",
            "source": source,
            "brief": session.get("last_user_brief", brief),
            "artboard_id": None,
            "node_ids": [],
            "round": 0,
            "retry_count": 0,
            "max_rounds": 3,
            "max_critic_retries": 1,
            "target_score": 8,
            "next_action": "end",
            "critique": {},
            "done": False,
            "html_used": "",
            "current_html": "",
            "create_new": True,
        }
        await self.event_broker.publish({"type": "generate_started", "source": source})
        try:
            result = await self.graph.ainvoke(initial_state)
            response = {
                "artboard_id": result.get("artboard_id"),
                "palette_artboard_id": session.get("palette_artboard_id"),
                "node_ids": result.get("node_ids", []),
                "html_used": result.get("html_used", ""),
                "round": result.get("round", 0),
                "critique": result.get("critique", {}),
                "done": result.get("done", False),
                "assistant_message": "I’ve started building from your brief and palette.",
                "questions": [],
                "conversation_stage": "building",
            }
            session["stage"] = "done"
            await self.event_broker.publish(
                {
                    "type": "generate_completed",
                    "source": source,
                    "round": response["round"],
                    "artboard_id": response["artboard_id"],
                    "done": response["done"],
                }
            )
            return response
        except Exception as exc:
            await self.event_broker.publish(
                {
                    "type": "generate_failed",
                    "source": source,
                    "message": str(exc),
                }
            )
            raise

    async def run_refine(self, artboard_id: str, instruction: str) -> dict[str, Any]:
        await self.event_broker.publish({"type": "refine_started", "artboard_id": artboard_id})
        current_html = await self._get_document_html(node_id=artboard_id)
        refined = await self._ask_designer(
            brief=instruction,
            current_html=current_html,
            critique={"suggestions": ["Apply the user instruction while preserving structure and visual quality."]},
        )

        write_result, mode_used = await self._apply_refine_html(artboard_id, refined["html"])
        node_ids = self._extract_created_node_ids(write_result)

        await self.event_broker.publish(
            {
                "type": "refine_completed",
                "artboard_id": artboard_id,
                "mode": mode_used,
                "node_ids": node_ids,
            }
        )

        return {
            "artboard_id": artboard_id,
            "node_ids": node_ids,
            "html_used": refined["html"],
            "tool_result": write_result,
            "mode_used": mode_used,
        }

    async def _handle_voice_turn(
        self,
        *,
        session_id: str,
        brief: str,
        source: Literal["text", "voice"],
        session: ConversationSession,
    ) -> dict[str, Any] | None:
        normalized_brief = brief.strip()
        session["last_user_brief"] = normalized_brief
        stage = session.get("stage", "intake")

        if stage == "intake":
            palette_artboard_id = session.get("palette_artboard_id")
            if not palette_artboard_id:
                palette_artboard_id = await self._create_palette_artboard(normalized_brief)
                session["palette_artboard_id"] = palette_artboard_id

            questions = [
                "How many pages do you need in the experience?",
                "Should I keep the palette focused on indigo and dark surfaces, or create 3 alternatives for you to approve first?",
            ]
            session["stage"] = "palette_requested"
            session["questions"] = questions
            session["palette_summary"] = "Three palette directions have been added to Paper canvas for review."

            await self.event_broker.publish(
                {
                    "type": "palette_artboard_created",
                    "conversation_id": session_id,
                    "artboard_id": palette_artboard_id,
                }
            )
            await self.event_broker.publish(
                {
                    "type": "clarifying_questions_sent",
                    "conversation_id": session_id,
                    "questions": questions,
                }
            )

            return {
                "artboard_id": None,
                "palette_artboard_id": palette_artboard_id,
                "node_ids": [],
                "html_used": "",
                "round": 0,
                "critique": {},
                "done": False,
                "assistant_message": "I’ve added three palette directions to Paper. Review and approve one palette, then I’ll start building.",
                "questions": questions,
                "conversation_stage": "palette_requested",
            }

        if stage == "palette_requested":
            if not self._is_palette_approval(normalized_brief):
                reminder_questions = session.get("questions") or [
                    "Tell me which palette you approve (Indigo Night, Amber Signal, or Aurora Glass).",
                    "Add one sentence for style direction so I can start building.",
                ]
                await self.event_broker.publish(
                    {
                        "type": "palette_approval_required",
                        "conversation_id": session_id,
                        "source": source,
                    }
                )
                return {
                    "artboard_id": None,
                    "palette_artboard_id": session.get("palette_artboard_id"),
                    "node_ids": [],
                    "html_used": "",
                    "round": 0,
                    "critique": {},
                    "done": False,
                    "assistant_message": "Palette approval is required before build. Reply with something like 'Approve Indigo Night and build the dashboard'.",
                    "questions": reminder_questions,
                    "conversation_stage": "palette_requested",
                }

            palette_context = await self._load_palette_context(session.get("palette_artboard_id"))
            if palette_context:
                session["palette_summary"] = self._summarize_palette_context(palette_context)
            session["stage"] = "building"
            await self.event_broker.publish(
                {
                    "type": "palette_reviewed",
                    "conversation_id": session_id,
                    "artboard_id": session.get("palette_artboard_id"),
                }
            )

        return None

    @staticmethod
    def _is_palette_approval(text: str) -> bool:
        lowered = text.lower()
        approval_tokens = [
            "approve",
            "approved",
            "go ahead",
            "looks good",
            "proceed",
            "use indigo",
            "use amber",
            "use aurora",
            "build it",
            "start build",
        ]
        return any(token in lowered for token in approval_tokens)

    async def get_current_canvas(self) -> dict[str, Any]:
        try:
            raw = await self._get_document_html(node_id=None)
            parsed = self._try_parse_json(raw)
            artboards: list[dict[str, Any]] = []
            if isinstance(parsed, dict):
                nodes = parsed.get("artboards") or parsed.get("nodes") or []
                if isinstance(nodes, list):
                    for node in nodes:
                        if isinstance(node, dict) and node.get("id"):
                            artboards.append({"id": str(node.get("id")), "name": str(node.get("name", "Untitled"))})

            return {
                "source": "get_document_html",
                "document_html": raw,
                "artboards": artboards,
            }
        except PaperMCPProtocolError:
            # Fallback for environments where get_document_html is not available.
            basic_info = await self.paper_client.invoke_tool("get_basic_info", {})
            payload = self._extract_primary_text_payload(basic_info)
            parsed = self._try_parse_json(payload)
            artboards: list[dict[str, Any]] = []
            if isinstance(parsed, dict):
                for artboard in parsed.get("artboards", []):
                    if isinstance(artboard, dict) and artboard.get("id"):
                        artboards.append({"id": str(artboard.get("id")), "name": str(artboard.get("name", "Untitled"))})

            return {
                "source": "get_basic_info",
                "document_html": "",
                "artboards": artboards,
                "raw": parsed,
            }

    async def _create_palette_artboard(self, brief: str) -> str:
        create_result = await self.paper_client.invoke_tool(
            "create_artboard",
            {
                "name": "Palette Exploration",
                "styles": {
                    "width": "1080px",
                    "height": "720px",
                    "backgroundColor": "#0a0a0a",
                    "display": "flex",
                    "flexDirection": "column",
                },
            },
        )
        palette_id = self._extract_node_id_from_create_artboard(create_result)
        if not palette_id:
            raise PaperMCPProtocolError("Failed to create palette exploration artboard.")

        palette_html = f"""
<div style="width:100%;height:100%;display:flex;flex-direction:column;gap:24px;padding:32px;background:linear-gradient(180deg, #0a0a0a 0%, #10162a 100%);color:#ffffff;font-family:Inter, sans-serif;">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:16px;">
    <div style="display:flex;flex-direction:column;gap:8px;max-width:720px;">
      <div style="font-size:12px;letter-spacing:0.18em;text-transform:uppercase;color:#a5b4fc;">Palette Direction</div>
      <div style="font-size:32px;font-weight:700;line-height:1.05;">Three color directions for the voice brief</div>
      <div style="font-size:16px;color:#cbd5e1;line-height:1.6;">{brief}</div>
    </div>
    <div style="padding:12px 16px;border-radius:999px;background:rgba(99,102,241,0.12);color:#c7d2fe;font-size:12px;">Tap a palette, then tweak it on canvas</div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;flex:1;">
    <div style="border-radius:24px;padding:20px;background:linear-gradient(180deg,#0f172a,#111827);border:1px solid rgba(148,163,184,0.16);display:flex;flex-direction:column;gap:14px;">
      <div style="display:flex;gap:10px;align-items:center;"><div style="width:12px;height:12px;border-radius:999px;background:#6366f1;"></div><div style="font-size:16px;font-weight:600;">Indigo Night</div></div>
      <div style="height:180px;border-radius:18px;background:linear-gradient(135deg,#020617,#0f172a 50%,#6366f1 130%);"></div>
      <div style="font-size:14px;color:#cbd5e1;line-height:1.6;">Deep navy, indigo, and cool slate for premium SaaS and finance.</div>
    </div>
    <div style="border-radius:24px;padding:20px;background:linear-gradient(180deg,#111111,#1b1b1f);border:1px solid rgba(244,114,182,0.16);display:flex;flex-direction:column;gap:14px;">
      <div style="display:flex;gap:10px;align-items:center;"><div style="width:12px;height:12px;border-radius:999px;background:#f59e0b;"></div><div style="font-size:16px;font-weight:600;">Amber Signal</div></div>
      <div style="height:180px;border-radius:18px;background:linear-gradient(135deg,#111111,#1f2937 50%,#f59e0b 130%);"></div>
      <div style="font-size:14px;color:#cbd5e1;line-height:1.6;">Warm accent, dark graphite surfaces, and strong conversion contrast.</div>
    </div>
    <div style="border-radius:24px;padding:20px;background:linear-gradient(180deg,#08131f,#0b1d2c);border:1px solid rgba(45,212,191,0.16);display:flex;flex-direction:column;gap:14px;">
      <div style="display:flex;gap:10px;align-items:center;"><div style="width:12px;height:12px;border-radius:999px;background:#22d3ee;"></div><div style="font-size:16px;font-weight:600;">Aurora Glass</div></div>
      <div style="height:180px;border-radius:18px;background:linear-gradient(135deg,#07111f,#0f172a 50%,#22d3ee 130%);"></div>
      <div style="font-size:14px;color:#cbd5e1;line-height:1.6;">Cool cyan, soft glass layers, and a lighter technological feel.</div>
    </div>
  </div>
</div>
""".strip()

        write_result = await self.paper_client.invoke_tool(
            "write_html",
            {
                "html": palette_html,
                "targetNodeId": palette_id,
                "mode": "insert-children",
            },
        )
        if write_result.get("isError"):
            message = self._extract_primary_text_payload(write_result)
            raise PaperMCPProtocolError(message or "Failed to write palette artboard HTML.")

        await self.event_broker.publish({"type": "palette_artboard_rendered", "artboard_id": palette_id})
        return palette_id

    async def _load_palette_context(self, palette_artboard_id: str | None) -> str:
        if not palette_artboard_id:
            return ""
        try:
            return await self._get_document_html(node_id=palette_artboard_id)
        except Exception:
            return ""

    @staticmethod
    def _summarize_palette_context(palette_context: str) -> str:
        lowered = palette_context.lower()
        if "indigo" in lowered or "#6366f1" in lowered:
            return "Indigo-forward palette confirmed from canvas."
        if "amber" in lowered or "#f59e0b" in lowered:
            return "Amber accent palette confirmed from canvas."
        if "cyan" in lowered or "#22d3ee" in lowered:
            return "Aurora glass palette confirmed from canvas."
        return "Palette updated from canvas."

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        graph.add_node("designer_node", self._designer_node)
        graph.add_node("critic_node", self._critic_node)
        graph.add_node("designer_refine_node", self._designer_refine_node)

        graph.add_edge(START, "designer_node")
        graph.add_edge("designer_node", "critic_node")
        graph.add_conditional_edges(
            "critic_node",
            self._should_continue,
            {
                "refine": "designer_refine_node",
                "retry_critic": "critic_node",
                "end": END,
            },
        )
        graph.add_edge("designer_refine_node", "critic_node")
        return graph.compile()

    async def _designer_node(self, state: PipelineState) -> PipelineState:
        await self.event_broker.publish({"type": "designer_started", "round": state.get("round", 0)})
        brief = state.get("brief", "")
        current_html = state.get("current_html", "")
        design = await self._ask_designer(brief=brief, current_html=current_html, critique={})

        create_payload = {
            "name": "Vibeframe App Canvas",
            "styles": {
                "width": "1440px",
                "height": "900px",
                "backgroundColor": "#0a0a0a",
                "display": "flex",
                "flexDirection": "column",
            },
        }
        create_result = await self.paper_client.invoke_tool("create_artboard", create_payload)
        artboard_id = self._extract_node_id_from_create_artboard(create_result)
        if not artboard_id:
            raise PaperMCPProtocolError("Failed to resolve artboard ID from create_artboard result.")

        write_result = await self.paper_client.invoke_tool(
            "write_html",
            {
                "html": design["html"],
                "targetNodeId": artboard_id,
                "mode": "insert-children",
            },
        )
        node_ids = self._extract_created_node_ids(write_result)

        await self.event_broker.publish(
            {
                "type": "designer_completed",
                "artboard_id": artboard_id,
                "node_ids": node_ids,
            }
        )

        return {
            **state,
            "status": "designing",
            "artboard_id": artboard_id,
            "node_ids": node_ids,
            "html_used": design["html"],
            "done": False,
        }

    async def _critic_node(self, state: PipelineState) -> PipelineState:
        artboard_id = state.get("artboard_id")
        if not artboard_id:
            raise PaperMCPProtocolError("Critic cannot run without an artboard_id.")

        current_html = await self._get_document_html(node_id=artboard_id)
        round_count = state.get("round", 0) + 1
        retry_count = state.get("retry_count", 0)
        try:
            critique = await self._ask_critic(
                brief=state.get("brief", ""),
                current_html=current_html,
                artboard_id=artboard_id,
            )
        except Exception as exc:
            max_critic_retries = state.get("max_critic_retries", 1)
            if retry_count < max_critic_retries:
                next_retry_count = retry_count + 1
                await self.event_broker.publish(
                    {
                        "type": "critic_retrying",
                        "round": round_count,
                        "retry": next_retry_count,
                        "message": str(exc),
                    }
                )
                return {
                    **state,
                    "status": "critiquing",
                    "current_html": current_html,
                    "retry_count": next_retry_count,
                    "round": round_count,
                    "next_action": "retry_critic",
                }

            critique = {
                "score": 10,
                "issues": ["Critic unavailable. Continuing with latest design revision."],
                "suggestions": ["Retry critique later when model/API is reachable."],
            }
            await self.event_broker.publish(
                {
                    "type": "critic_degraded",
                    "round": round_count,
                    "message": str(exc),
                }
            )
            return {
                **state,
                "status": "degraded",
                "current_html": current_html,
                "critique": critique,
                "retry_count": retry_count,
                "round": round_count,
                "next_action": "end",
                "done": True,
            }

        next_action = self._resolve_next_action(
            score=critique.get("score", 0),
            round_count=round_count,
            max_rounds=state.get("max_rounds", 3),
            target_score=state.get("target_score", 8),
        )

        await self.event_broker.publish(
            {
                "type": "critic_completed",
                "round": round_count,
                "score": critique.get("score", 0),
                "issues": critique.get("issues", []),
                "next_action": next_action,
            }
        )

        return {
            **state,
            "status": "critiquing",
            "current_html": current_html,
            "critique": critique,
            "retry_count": 0,
            "round": round_count,
            "next_action": next_action,
            "done": next_action == "end",
        }

    async def _designer_refine_node(self, state: PipelineState) -> PipelineState:
        artboard_id = state.get("artboard_id")
        if not artboard_id:
            raise PaperMCPProtocolError("Refinement cannot run without an artboard_id.")

        await self.event_broker.publish({"type": "designer_refine_started", "round": state.get("round", 0)})

        current_html = await self._get_document_html(node_id=artboard_id)
        design = await self._ask_designer(
            brief=state.get("brief", ""),
            current_html=current_html,
            critique=state.get("critique", {}),
        )

        write_result, mode_used = await self._apply_refine_html(artboard_id, design["html"])
        node_ids = self._extract_created_node_ids(write_result)

        await self.event_broker.publish(
            {
                "type": "designer_refine_completed",
                "artboard_id": artboard_id,
                "mode": mode_used,
                "node_ids": node_ids,
            }
        )

        return {
            **state,
            "status": "refining",
            "node_ids": node_ids,
            "html_used": design["html"],
        }

    def _should_continue(self, state: PipelineState) -> str:
        action = state.get("next_action", "end")
        if action == "retry_critic":
            return "retry_critic"
        if action == "refine":
            return "refine"
        state["done"] = True
        state["status"] = "complete"
        return "end"

    @staticmethod
    def _resolve_next_action(
        *, score: Any, round_count: int, max_rounds: int, target_score: int
    ) -> Literal["refine", "end"]:
        numeric_score: float
        if isinstance(score, (int, float)):
            numeric_score = float(score)
        else:
            numeric_score = 0.0

        if round_count < max_rounds and numeric_score < float(target_score):
            return "refine"
        return "end"

    async def _ask_designer(self, brief: str, current_html: str, critique: dict[str, Any]) -> dict[str, str]:
        critique_json = json.dumps(critique or {}, ensure_ascii=True)
        current_html_hint = current_html or "<empty-canvas />"
        message = (
            f"User brief:\n{brief}\n\n"
            f"Current canvas HTML:\n{current_html_hint}\n\n"
            f"Critique to address:\n{critique_json}\n\n"
            "Generate complete landing page HTML with all required sections and inline styles."
        )

        response = await self.designer_llm.ainvoke(
            [
                SystemMessage(content=DESIGNER_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ]
        )
        payload = self._try_parse_json(getattr(response, "content", ""))
        if not isinstance(payload, dict):
            raise ValueError("Designer output was not a JSON object.")

        summary = str(payload.get("summary", "Generated rich landing page."))
        html = str(payload.get("html", "")).strip()
        if not html:
            raise ValueError("Designer output did not contain html.")

        return {
            "summary": summary,
            "html": html,
        }

    async def _ask_critic(self, brief: str, current_html: str, artboard_id: str) -> dict[str, Any]:
        screenshot = await self._get_artboard_screenshot_payload(artboard_id)

        if self.gemini_api_key:
            try:
                critique = await self._ask_critic_with_gemini(brief=brief, current_html=current_html, screenshot=screenshot)
                await self.event_broker.publish(
                    {
                        "type": "critic_provider_used",
                        "provider": "gemini",
                        "model": self.gemini_critic_model,
                        "used_screenshot": bool(screenshot),
                    }
                )
                return self._normalize_critique_payload(critique)
            except Exception as exc:
                await self.event_broker.publish(
                    {
                        "type": "critic_provider_fallback",
                        "from": "gemini",
                        "to": "groq",
                        "message": str(exc),
                    }
                )

        message = (
            f"User brief:\n{brief}\n\n"
            f"Current canvas HTML:\n{current_html}\n\n"
            "Provide score, issues, and suggestions as strict JSON."
        )

        response = await self.critic_llm.ainvoke(
            [
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ]
        )
        payload = self._try_parse_json(getattr(response, "content", ""))
        if not isinstance(payload, dict):
            raise ValueError("Critic output was not a JSON object.")

        await self.event_broker.publish(
            {
                "type": "critic_provider_used",
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
                "used_screenshot": False,
            }
        )

        return self._normalize_critique_payload(payload)

    async def _ask_critic_with_gemini(
        self,
        *,
        brief: str,
        current_html: str,
        screenshot: tuple[str, str] | None,
    ) -> dict[str, Any]:
        url = f"{self.gemini_api_base}/models/{self.gemini_critic_model}:generateContent"
        parts: list[dict[str, Any]] = [
            {
                "text": (
                    f"User brief:\n{brief}\n\n"
                    "Critique the current design visually with strict JSON output only."
                    " If screenshot is unavailable, use HTML cues and report lower confidence.\n\n"
                    f"Current canvas HTML:\n{current_html}"
                )
            }
        ]

        if screenshot:
            mime_type, image_b64 = screenshot
            parts.append({"inline_data": {"mime_type": mime_type, "data": image_b64}})

        payload = {
            "systemInstruction": {"parts": [{"text": CRITIC_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(url, params={"key": self.gemini_api_key}, json=payload)
            response.raise_for_status()
            body = response.json()

        candidates = body.get("candidates", []) if isinstance(body, dict) else []
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Gemini critic response did not include candidates.")

        candidate = candidates[0] if isinstance(candidates[0], dict) else {}
        content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
        parts = content.get("parts", []) if isinstance(content, dict) else []
        text_payload = ""
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_payload = part["text"]
                    break

        parsed = self._try_parse_json(text_payload)
        if not isinstance(parsed, dict):
            raise ValueError("Gemini critic output was not valid JSON.")

        return parsed

    @staticmethod
    def _normalize_critique_payload(payload: dict[str, Any]) -> dict[str, Any]:

        score = payload.get("score", 0)
        try:
            normalized_score = int(float(score))
        except (TypeError, ValueError):
            normalized_score = 0

        issues = payload.get("issues", [])
        suggestions = payload.get("suggestions", [])
        if not isinstance(issues, list):
            issues = []
        if not isinstance(suggestions, list):
            suggestions = []

        return {
            "score": max(0, min(10, normalized_score)),
            "issues": [str(issue) for issue in issues],
            "suggestions": [str(item) for item in suggestions],
        }

    async def _get_artboard_screenshot_payload(self, artboard_id: str) -> tuple[str, str] | None:
        try:
            result = await self.paper_client.invoke_tool("get_screenshot", {"nodeId": artboard_id})
        except Exception:
            return None

        if result.get("isError"):
            return None

        content = result.get("content", [])
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") == "image":
                    image_payload = self._extract_image_payload(item)
                    if image_payload:
                        return image_payload

                if item.get("type") == "text":
                    text = item.get("text")
                    parsed = self._try_parse_json(text)
                    if parsed is not None:
                        image_payload = self._extract_image_payload(parsed)
                        if image_payload:
                            return image_payload

        image_payload = self._extract_image_payload(result)
        return image_payload

    @staticmethod
    def _extract_image_payload(value: Any) -> tuple[str, str] | None:
        if isinstance(value, dict):
            for key in ("dataUrl", "data_url", "imageDataUrl", "image_data_url"):
                data_url = value.get(key)
                if isinstance(data_url, str):
                    parsed = VibeframeAgentPipeline._parse_data_url(data_url)
                    if parsed:
                        return parsed

            for key in ("base64", "imageBase64", "image_base64", "data"):
                raw = value.get(key)
                if isinstance(raw, str) and raw.strip():
                    mime = str(value.get("mimeType") or value.get("mime_type") or "image/png")
                    candidate = raw.strip()
                    parsed = VibeframeAgentPipeline._parse_data_url(candidate)
                    if parsed:
                        return parsed
                    return mime, candidate

            for nested in value.values():
                nested_payload = VibeframeAgentPipeline._extract_image_payload(nested)
                if nested_payload:
                    return nested_payload

        if isinstance(value, list):
            for item in value:
                nested_payload = VibeframeAgentPipeline._extract_image_payload(item)
                if nested_payload:
                    return nested_payload

        if isinstance(value, str):
            parsed = VibeframeAgentPipeline._parse_data_url(value)
            if parsed:
                return parsed

        return None

    @staticmethod
    def _parse_data_url(value: str) -> tuple[str, str] | None:
        if not value.startswith("data:"):
            return None
        if ";base64," not in value:
            return None

        header, payload = value.split(",", 1)
        mime_type = header[5:].split(";", 1)[0].strip() or "image/png"
        if not payload.strip():
            return None
        return mime_type, payload.strip()

    async def _get_document_html(self, node_id: str | None) -> str:
        args: dict[str, Any] = {}
        if node_id:
            args["nodeId"] = node_id

        result = await self.paper_client.invoke_tool("get_document_html", args)
        if result.get("isError"):
            message = self._extract_primary_text_payload(result)
            if "Unknown tool: get_document_html" in message:
                if node_id:
                    jsx_result = await self.paper_client.invoke_tool(
                        "get_jsx",
                        {
                            "nodeId": node_id,
                            "format": "inline-styles",
                        },
                    )
                    return self._extract_primary_text_payload(jsx_result)

                info_result = await self.paper_client.invoke_tool("get_basic_info", {})
                return self._extract_primary_text_payload(info_result)

            raise PaperMCPProtocolError(message or "get_document_html returned error.")

        return self._extract_primary_text_payload(result)

    async def _apply_refine_html(self, artboard_id: str, html: str) -> tuple[dict[str, Any], str]:
        update_result = await self.paper_client.invoke_tool(
            "write_html",
            {
                "html": html,
                "targetNodeId": artboard_id,
                "mode": "update",
            },
        )
        if not update_result.get("isError"):
            return update_result, "update"

        update_error = self._extract_primary_text_payload(update_result)
        update_error_lower = update_error.lower()
        allow_fallback = (
            "mode" in update_error_lower
            or "unsupported" in update_error_lower
            or "insert-children" in update_error_lower
            or "unknown" in update_error_lower
        )
        if not allow_fallback:
            raise PaperMCPProtocolError(update_error or "write_html update failed during refine.")

        await self.event_broker.publish(
            {
                "type": "refine_update_fallback",
                "artboard_id": artboard_id,
                "reason": update_error or "update mode unsupported",
            }
        )

        existing_children = await self._get_child_node_ids(artboard_id)
        if existing_children:
            delete_result = await self.paper_client.invoke_tool("delete_nodes", {"nodeIds": existing_children})
            if delete_result.get("isError"):
                message = self._extract_primary_text_payload(delete_result)
                raise PaperMCPProtocolError(message or "delete_nodes failed while preparing refine.")

        write_result = await self.paper_client.invoke_tool(
            "write_html",
            {
                "html": html,
                "targetNodeId": artboard_id,
                "mode": "insert-children",
            },
        )
        if write_result.get("isError"):
            message = self._extract_primary_text_payload(write_result)
            raise PaperMCPProtocolError(message or "write_html refine call failed.")

        return write_result, "insert-children"

    async def _get_child_node_ids(self, node_id: str) -> list[str]:
        result = await self.paper_client.invoke_tool("get_children", {"nodeId": node_id})
        if result.get("isError"):
            message = self._extract_primary_text_payload(result)
            raise PaperMCPProtocolError(message or "get_children failed.")

        payload = self._try_parse_json(self._extract_primary_text_payload(result))
        child_ids: list[str] = []
        if isinstance(payload, dict):
            children = payload.get("children", [])
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        child_id = child.get("id")
                        if isinstance(child_id, str) and child_id:
                            child_ids.append(child_id)
        return child_ids

    @staticmethod
    def _extract_primary_text_payload(result: dict[str, Any]) -> str:
        content = result.get("content", [])
        if not isinstance(content, list):
            return ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    return text
        return ""

    @staticmethod
    def _extract_node_id_from_create_artboard(result: dict[str, Any]) -> str | None:
        text = VibeframeAgentPipeline._extract_primary_text_payload(result)
        parsed = VibeframeAgentPipeline._try_parse_json(text)
        if isinstance(parsed, dict):
            node_id = parsed.get("id")
            if isinstance(node_id, str) and node_id:
                return node_id
        return None

    @staticmethod
    def _extract_created_node_ids(result: dict[str, Any]) -> list[str]:
        text = VibeframeAgentPipeline._extract_primary_text_payload(result)
        parsed = VibeframeAgentPipeline._try_parse_json(text)
        if isinstance(parsed, dict):
            created = parsed.get("createdNodes")
            if isinstance(created, list):
                return [str(item.get("id")) for item in created if isinstance(item, dict) and item.get("id")]
        return []

    @staticmethod
    def _try_parse_json(raw: Any) -> Any:
        if isinstance(raw, dict) or isinstance(raw, list):
            return raw
        if not isinstance(raw, str):
            return None

        text = raw.strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None
