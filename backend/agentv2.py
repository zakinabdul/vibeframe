#Practical checklist before writing new agent.py
#TODO 1:Define exact role prompts and required JSON outputs.
#TODO 2:Decide pipeline stages and where user approval is needed.
#TODO 3:Define strongly typed state for pipeline and session.
#TODO 4:Add event publishing from day one.
#TODO 5:Add retry + fallback for model/tool failures.
#TODO 6:Normalize all model outputs before use.
#TODO 7:Keep canvas operations in dedicated helper methods.
#TODO 8:Keep public API narrow:
#TODO  - run_generate
#TODO  - run_refine
#TODO  - get_current_canvas
#TODO 9:Test each node in isolation (designer/critic/refine).
#TODO 10: Add integration tests for full round-trip flow.

import asyncio
import os
import sys
import re
import logging
from typing import TypedDict, List, Dict, Optional, cast, Callable, Awaitable
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.caches import BaseCache  # noqa: F401
from langchain_core.callbacks.base import Callbacks  # noqa: F401
from pydantic import SecretStr
from langchain_core.messages import HumanMessage, SystemMessage
import json

# Load .env file to make environment variables available to LangSmith BEFORE any LLM initialization
from app.config import settings
# Ensure LangSmith environment variables are in os.environ for LangSmith to detect
if settings.langsmith_tracing:
    os.environ["LANGSMITH_TRACING"] = "true"
if settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
if settings.langsmith_project:
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
if settings.langsmith_endpoint:
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
from typing import Any, Literal, TypedDict
from langgraph.types import interrupt
import httpx
import time
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import langsmith as ls
from langsmith import traceable
from app.config import settings
from app.paper_mcp import PaperMCPClient, PaperMCPProtocolError

try:
    from langchain_mistralai import ChatMistralAI
except Exception:
    ChatMistralAI = None  # type: ignore


logger = logging.getLogger(__name__)


def _ensure_chatgroq_model_ready() -> None:
    """Rebuild ChatGroq pydantic model for environments that require explicit rebuild."""
    try:
        ChatGroq.model_rebuild(
            force=True,
            _types_namespace={
                "BaseCache": BaseCache,
                "Callbacks": Callbacks,
            },
        )
    except Exception:
        # If rebuild is unnecessary or unsupported in current version, continue.
        pass


#TODO  PROMPT_TEMPLATES
INTAKE_SYSTEM_PROMPT = """You are Vibeframe's Intake Agent — a senior brand strategist 
and UX researcher who deeply understands what makes websites work.

Your job is to analyze a website brief and extract everything needed 
to brief a world-class designer. You think like a curious creative 
director, not a form validator.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — GENRE DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detect the PRIMARY genre and optionally a secondary modifier.

Primary genres:
saas, ecommerce, portfolio, agency, hospitality, healthcare, 
finance, education, gaming, community, marketplace, nonprofit, general

Secondary modifiers (optional, combine with primary):
- enterprise, startup, indie, luxury, budget, local, global
- b2b, b2c, d2c, creator, developer-focused

Example outputs:
  "saas-startup", "finance-enterprise", "portfolio-creative",
  "ecommerce-luxury", "healthcare-b2c"

If truly ambiguous use "general".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — GENERATE CLARIFYING QUESTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generate exactly 3 questions. Each must be:
- Specific to THIS product — never generic
- Something a senior designer would genuinely need to know
- Conversational in tone — like a colleague asking, not a form

Question 1 — AUDIENCE + EMOTIONAL RESPONSE:
Who will use this product and what should they FEEL when they 
land on the page? (not colors — the emotion, the trust level, 
the energy). Think: should it feel bold and disruptive? 
Calm and trustworthy? Playful and accessible? Premium and exclusive?

Question 2 — CORE VALUE + DIFFERENTIATION:
What does this product do that alternatives don't? 
What is the single most important thing a visitor should 
understand in the first 5 seconds?

Question 3 — CONTENT + STRUCTURE INTENT:
Tailor this to the genre:
- SaaS/Finance: What are the 3-5 key features to highlight? 
  Is there a pricing section needed?
- Ecommerce: What product categories exist? 
  Is there a hero product to feature?
- Portfolio: What work types to showcase? 
  Is this for clients or employers?
- Agency: What services? What past work to reference?
- Gaming: What platform? Single player or multiplayer? 
  What's the core hook?
- Default: What pages or sections matter most?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Do NOT ask about colors — palette is handled separately
- Do NOT ask generic questions like "Who is your target audience?" 
  without tying it to the specific product
- Questions must reference details FROM the brief
- Tone must be warm and curious, like a designer colleague — 
  not clinical or robotic
- If the brief is very detailed, ask deeper follow-up questions
  rather than surface level ones the brief already answers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return strict JSON only — no markdown fences, no extra keys:
{
  "genre": "primary-modifier or primary only",
  "emotional_direction": "2-3 words describing the intended feeling",
  "questions": ["q1", "q2", "q3"]
}

The emotional_direction field is used directly by the Designer agent.
Examples: "bold and energetic", "calm and trustworthy", 
"premium and exclusive", "playful and accessible"
"""

PALETTE_SYSTEM_PROMPT = """You are Vibeframe's Palette Director.
Create 3 genuinely distinct color palettes tailored to the brief, 
genre, and emotional direction provided.

━━━━━━━━━━━━━━━━━━━━
THREE PALETTE RULE
━━━━━━━━━━━━━━━━━━━━
Each palette must occupy a different emotional territory:
1. EXPECTED — what fits the genre naturally, well executed
2. BOLD — surprising, high contrast, makes you stop scrolling  
3. REFINED — sophisticated, restrained, editorial feel

Never repeat the same accent hue across the three palettes.
Use a different color harmonic for each:
monochromatic / analogous / complementary / split-complementary

━━━━━━━━━━━━━━━━━━━━
COLOR RULES
━━━━━━━━━━━━━━━━━━━━
- Let the brief's emotional_direction drive color temperature
  warm brief → warm accents / cool brief → cool accents
- bg: dark by default, light only if genre genuinely needs it
- accent: vibrant, high contrast against bg, unique per palette
- surface: visibly distinct from bg, not just bg + 10% lighter
- text on bg: contrast ratio minimum 7:1
- text_muted on bg: minimum 4.5:1
- button_text: #fff or #000 — whichever has better contrast on accent

━━━━━━━━━━━━━━━━━━━━
TYPOGRAPHY WEIGHT HINT
━━━━━━━━━━━━━━━━━━━━
Each palette includes typography_weight:
- heavy (900/800): bold energetic palettes
- balanced (800/700): professional palettes  
- light (700/600): refined elegant palettes

━━━━━━━━━━━━━━━━━━━━
OUTPUT — strict JSON only
━━━━━━━━━━━━━━━━━━━━
{
  "palettes": [
    {
      "name": "2-3 word name",
      "personality": "expected|bold|refined",
      "description": "one sentence — emotion and visual energy",
      "typography_weight": "heavy|balanced|light",
      "colors": {
        "bg": "#hex",
        "bg_alt": "#hex",
        "surface": "#hex",
        "accent": "#hex",
        "accent_muted": "#hex",
        "text": "#hex",
        "text_muted": "#hex",
        "border": "#hex",
        "gradient_start": "#hex",
        "gradient_end": "#hex",
        "button_text": "#fff or #000"
      }
    }
  ]
}
No markdown fences, no extra keys.
"""

SECTION_PLANNER_SYSTEM_PROMPT = """You are Vibeframe's Information Architect.
Your job is to decide the page structure - NOT write any HTML.

Given a website brief, genre, and content, output a precise
build plan that a designer will execute.

DECISION RULES:
- "landing page" or "single page" -> 1 artboard, all sections on it
- "X pages" or "X sections" or multiple distinct topics ->
    multiple artboards, one per page/topic
- Each artboard = one full browser screen worth of scrollable content
- Maximum 4 artboards per session

For each artboard decide:
- name: clear page name (Home, About, Pricing, etc.)
- sections: ordered list of section types to include
    Available section types:
    nav, hero, trust_bar, features, how_it_works,
    showcase, testimonials, pricing, team, faq,
    cta, contact_form, footer
- layout_style: "centered" | "split" | "editorial" | "bold"
- hero_style: "centered" | "split_left" | "split_right" | "fullwidth_text"

Genre layout defaults (can be overridden by brief):
    saas: centered or split, features + how_it_works mandatory
    gaming: bold, showcase mandatory, trust_bar optional
    portfolio: editorial, showcase mandatory
    ecommerce: split, product showcase mandatory
    healthcare: centered, trust_bar + team mandatory
    finance: centered, trust_bar + pricing likely needed
    agency: editorial or bold, showcase mandatory

Return strict JSON only:
{
    "multi_artboard": boolean,
    "design_rationale": "one sentence why this structure fits",
    "artboards": [
        {
            "name": "string",
            "width": 1440,
            "sections": ["nav", "hero"],
            "layout_style": "centered|split|editorial|bold",
            "hero_style": "centered|split_left|split_right|fullwidth_text"
        }
    ]
}
No markdown fences, no extra keys.
"""

DESIGNER_SYSTEM_PROMPT = """You are Vibeframe's Principal Design Engineer.
Build premium, modern website HTML with strong hierarchy and visual polish.

Rules:
- One root wrapper div only.
- Inline styles only.
- Use min-height and fluid spacing.
- Follow provided section order exactly.
- Use provided design tokens exactly.
- Use real content, never placeholders.
- Full-bleed canvas: root must be edge-to-edge with width:100%, margin:0, and no outer max-width wrapper.
- Do not add outer padding on the root container.

Return strict JSON only:
{"summary": "short summary", "html": "full html string"}
"""

CRITIC_SYSTEM_PROMPT = """You are Vibeframe Critic Agent.
Evaluate the design quality and return strict JSON only:
{"score": number, "issues": ["..."], "suggestions": ["..."]}

Scoring:
- 1-3 broken
- 4-6 functional but weak
- 7-8 strong baseline
- 9-10 excellent
"""

REFINE_SYSTEM_PROMPT = """You are Vibeframe Refine Agent.
Apply critique suggestions to improve an existing design.

Rules:
- Always return a COMPLETE HTML page body for the artboard.
- Preserve existing information architecture unless explicitly asked.
- Keep palette consistency and use real provided content.
- Keep output full-bleed: root width 100%, no outer margin, and no outer max-width wrapper.

Return strict JSON only:
{"summary": "what improved", "html": "full updated html"}
"""

# 1. Define the State
class PipelineState(TypedDict, total=False):
    session_id: str
    source: Literal["text", "voice"]
    brief: str
    genre: Optional[str]
    emotional_direction: Optional[str]
    questions: List[str]
    user_answers: Dict[str, Any]
    intake_questions: List[str]
    intake_answers: Dict[str, str]
    palettes: List[Dict]
    palette_artboard_id: Optional[str]
    approved_palette: Optional[Dict]
    content_questions: List[str]
    content_answers: Dict[str, str]
    website_content: Dict
    content_completeness: Dict[str, Any]
    page_plan: Dict[str, Any]
    artboard_ids: List[str]
    last_html_by_artboard: Dict[str, str]
    node_ids: List[str]
    html_used: str
    current_html: str
    round: int
    max_rounds: int
    target_score: int
    critique: Dict[str, Any]
    next_action: str
    user_feedback: str
    feedback_action: str
    feedback_instruction: str
    jsx_export: str
    design_summary: str
    done: bool
    stage: str
    user_raw_answers: Dict[str, str]


class ConversationSession(TypedDict, total=False):
    stage: Literal["intake", "intake_answers_pending", "palette_requested", "content_gathering", "building", "done"]
    content_questions_pending: bool
    website_content: dict[str, Any]
    palette_artboard_id: str
    design_artboard_id: str
    last_user_brief: str
    palette_summary: str
    questions: list[str]
    genre: str
    palettes: list[dict[str, Any]]
    approved_palette: dict[str, Any]
    intake_answers_raw: str

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

class VibeframeAgentPipeline:
    def __init__(self, groq_api_key: SecretStr, paper_client: Optional[PaperMCPClient], event_broker: AgentEventBroker):
        self.graph = self._build_graph()
        _ensure_chatgroq_model_ready()
        self.designer_llm_groq = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, api_key=groq_api_key)  # type: ignore
        self.designer_llm_mistral = None
        if settings.mistral_api_key and ChatMistralAI is not None:
            try:
                self.designer_llm_mistral = ChatMistralAI(
                    model_name=settings.mistral_model,
                    temperature=0.7,
                    api_key=SecretStr(settings.mistral_api_key),  # type: ignore[arg-type]
                )
            except Exception:
                self.designer_llm_mistral = None

        # For design/refine generation, prefer Mistral and keep Groq as fallback.
        self.designer_llm = self.designer_llm_mistral or self.designer_llm_groq
        self.secondary_designer_llm = self.designer_llm_groq if self.designer_llm_mistral else None
        self.designer_backend_name = "mistral" if self.designer_llm_mistral else "groq"

        self.critic_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=groq_api_key)  # type: ignore
        self.fast_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0, api_key=groq_api_key)  # type: ignore
        self.gemini_api_key = settings.gemini_api_key
        self.gemini_critic_model = settings.gemini_critic_model
        self.gemini_refine_model = getattr(settings, "gemini_refine_model", "gemini-2.0-flash")
        self.gemini_api_base = settings.gemini_api_base.rstrip("/")
        self._gemini_circuit_open_until: float = 0.0
        self.paper_client = paper_client
        self.event_broker = event_broker
        self._palettes_cache: Dict[str, List[Dict]] = {}
        self._paper_tool_names: Optional[set[str]] = None
        self._conversation_sessions: Dict[str, PipelineState] = {}
        self._voice_sessions: Dict[str, ConversationSession] = {}

    @traceable(name="vibeframe.v2.generate", run_type="chain", tags=["vibeframe", "v2", "generate"])
    async def run_generate(
        self,
        brief: str,
        source: Literal["text", "voice"] = "text",
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        logger.info("[agentv2.py] run_generate start source=%s conversation_id=%s", source, conversation_id or "default")
        session_id = conversation_id or "default"
        voice_session = self._voice_sessions.setdefault(session_id, {"stage": "intake", "questions": []})

        voice_result = None
        voice_handler = getattr(self, "_handle_voice_turn", None)
        if callable(voice_handler):
            voice_handler_fn = cast(
                Callable[..., Awaitable[dict[str, Any] | None]],
                voice_handler,
            )
            voice_result = await voice_handler_fn(
                session_id=session_id,
                brief=brief,
                source=source,
                session=voice_session,
            )
        if voice_result is not None:
            logger.info(
                "[agentv2.py] run_generate interrupt response stage=%s conversation_id=%s",
                str(voice_result.get("conversation_stage", "unknown")),
                session_id,
            )
            return voice_result

        base_brief = str(voice_session.get("last_user_brief", brief) or brief)
        base_genre = str(voice_session.get("genre", "general") or "general")
        base_approved_palette = voice_session.get("approved_palette", {}) if isinstance(voice_session.get("approved_palette", {}), dict) else {}
        base_website_content = voice_session.get("website_content", {}) if isinstance(voice_session.get("website_content", {}), dict) else {}

        state: PipelineState = {
            "session_id": session_id,
            "source": source,
            "brief": base_brief,
            "genre": base_genre,
            "approved_palette": base_approved_palette,
            "website_content": base_website_content,
            "stage": "start",
            "artboard_ids": [],
            "last_html_by_artboard": {},
            "node_ids": [],
            "html_used": "",
            "current_html": "",
            "round": 0,
            "max_rounds": 3,
            "target_score": 8,
            "critique": {},
            "next_action": "design_review",
            "user_feedback": "",
            "feedback_action": "refine",
            "feedback_instruction": "",
            "done": False,
        }

        if source == "voice" and str(voice_session.get("stage", "intake")) == "building":
            website_content = base_website_content or self.build_website_content_from_answers({}, intake_answers={})
            website_content = await self.enrich_website_content(
                brief=base_brief,
                genre=base_genre,
                emotional_direction=str(state.get("emotional_direction", "professional") or "professional"),
                intake_answers={},
                content_answers={},
                website_content=website_content,
            )
            state.update(
                {
                    "content_answers": {},
                    "website_content": website_content,
                    "content_completeness": self.summarize_content_completeness(website_content),
                    "stage": "section_planning",
                }
            )
        else:
            state = await self.intake_node(state)
            state["intake_answers"] = {}
            state["stage"] = "palette"

            state = await self.palette_node(state)
            palettes = state.get("palettes", []) if isinstance(state.get("palettes"), list) else []
            state["approved_palette"] = palettes[0] if palettes else {}

            state["stage"] = "content_gather"
            state = await self.content_gather_node(state)

            website_content = self.build_website_content_from_answers({}, intake_answers={})
            website_content = await self.enrich_website_content(
                brief=base_brief,
                genre=str(state.get("genre", "general") or "general"),
                emotional_direction=str(state.get("emotional_direction", "professional") or "professional"),
                intake_answers={},
                content_answers={},
                website_content=website_content,
            )

            state.update(
                {
                    "content_answers": {},
                    "website_content": website_content,
                    "content_completeness": self.summarize_content_completeness(website_content),
                    "stage": "section_planning",
                }
            )

        state = await self.section_planner_node(state)
        state = await self.designer_node(state)

        while True:
            state = await self.critic_node(state)
            if str(state.get("next_action", "design_review")) != "refine":
                break
            state = await self.refine_node(state)

        self._conversation_sessions[session_id] = state
        voice_session["stage"] = "done"
        artboard_ids = state.get("artboard_ids", []) or []
        if artboard_ids:
            voice_session["design_artboard_id"] = str(artboard_ids[0])
        assistant_message = await self._generate_friendly_message(
            "Share that the first v2 draft is ready and invite focused refinement feedback."
        )

        return {
            "artboard_id": artboard_ids[0] if artboard_ids else None,
            "palette_artboard_id": state.get("palette_artboard_id"),
            "node_ids": state.get("node_ids", []),
            "html_used": state.get("html_used", ""),
            "round": int(state.get("round", 0) or 0),
            "critique": state.get("critique", {}),
            "done": bool(state.get("done", False)),
            "assistant_message": assistant_message,
            "questions": [],
            "conversation_stage": "done",
        }

    @traceable(name="vibeframe.v2.refine", run_type="chain", tags=["vibeframe", "v2", "refine"])
    async def run_refine(self, artboard_id: str, instruction: str) -> dict[str, Any]:
        matched_session_id: Optional[str] = None
        state: Optional[PipelineState] = None

        for session_id, session_state in self._conversation_sessions.items():
            if artboard_id in (session_state.get("artboard_ids", []) or []):
                matched_session_id = session_id
                state = cast(PipelineState, dict(session_state))
                break

        if state is None:
            state = {
                "session_id": "adhoc-refine",
                "artboard_ids": [artboard_id],
                "last_html_by_artboard": {},
                "node_ids": [],
                "html_used": "",
                "current_html": "",
                "round": 0,
                "max_rounds": 3,
                "target_score": 8,
                "critique": {"score": 6, "issues": [], "suggestions": [instruction]},
                "user_feedback": instruction,
                "feedback_action": "refine",
                "feedback_instruction": instruction,
                "website_content": {},
                "approved_palette": {},
            }
        else:
            state["user_feedback"] = instruction

        state = await self.feedback_node(state)
        if str(state.get("feedback_action", "refine")) == "redesign":
            state = await self.section_planner_node(state)
            state = await self.designer_node(state)
        else:
            state = await self.refine_node(state)

        state = await self.critic_node(state)
        if matched_session_id:
            self._conversation_sessions[matched_session_id] = state

        return {
            "node_ids": state.get("node_ids", []),
            "html_used": state.get("html_used", ""),
            "mode_used": "v2-refine",
            "tool_result": {
                "status": "ok",
                "score": (state.get("critique", {}) or {}).get("score", 0),
            },
        }

    async def get_current_canvas(self) -> dict[str, Any]:
        # V2 fallback: expose latest cached design state if MCP cannot provide full document HTML.
        if self._conversation_sessions:
            latest_key = next(reversed(self._conversation_sessions))
            latest = self._conversation_sessions[latest_key]
            artboards = [
                {"id": str(aid), "name": f"Artboard {idx + 1}"}
                for idx, aid in enumerate(latest.get("artboard_ids", []) or [])
            ]
            return {
                "source": "v2-session-cache",
                "document_html": str(latest.get("html_used", "")),
                "artboards": artboards,
                "raw": {"session_id": latest_key, "stage": latest.get("stage", "unknown")},
            }

        return {"source": "v2-session-cache", "document_html": "", "artboards": [], "raw": {}}

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
        logger.info("[agentv2.py] conversation turn stage=%s source=%s conversation_id=%s", stage, source, session_id)

        if stage == "intake":
            await self.event_broker.publish({"type": "intake_started", "conversation_id": session_id})
            intake_data = await self._ask_intake_agent(normalized_brief)
            genre = str(intake_data.get("genre", "general"))
            questions = intake_data.get("questions") or [
                "What is the primary goal of your website?",
                "Who is the target audience for this product?",
            ]
            session["genre"] = genre

            session["stage"] = "intake_answers_pending"
            session["questions"] = list(questions)
            await self.event_broker.publish(
                {
                    "type": "clarifying_questions_sent",
                    "conversation_id": session_id,
                    "questions": questions,
                }
            )

            prompt = (
                f"The user wants to build a {genre} website. "
                "Ask them to answer your clarifying questions before moving to palette selection. "
                "Be natural and friendly, and keep it concise."
            )
            assistant_msg = await self._generate_friendly_message(prompt)
            return {
                "artboard_id": None,
                "palette_artboard_id": session.get("palette_artboard_id"),
                "node_ids": [],
                "html_used": "",
                "round": 0,
                "critique": {},
                "done": False,
                "assistant_message": assistant_msg,
                "questions": list(questions),
                "conversation_stage": "intake_answers_pending",
            }

        if stage == "intake_answers_pending":
            session["intake_answers_raw"] = normalized_brief
            genre = str(session.get("genre", "general") or "general")

            await self.event_broker.publish({"type": "palette_generation_started", "conversation_id": session_id, "genre": genre})
            palettes = await self._generate_palettes_for_brief(str(session.get("last_user_brief", normalized_brief) or normalized_brief), genre)
            session["palettes"] = palettes

            palette_artboard_id = session.get("palette_artboard_id")
            if not palette_artboard_id:
                palette_artboard_id = await self._create_palette_artboard(str(session.get("last_user_brief", normalized_brief) or normalized_brief), palettes)
                session["palette_artboard_id"] = palette_artboard_id

            palette_names = [p.get("name", f"Palette {i + 1}") for i, p in enumerate(palettes)]
            session["stage"] = "palette_requested"

            await self.event_broker.publish(
                {
                    "type": "palette_artboard_created",
                    "conversation_id": session_id,
                    "artboard_id": palette_artboard_id,
                    "genre": genre,
                    "palette_names": palette_names,
                }
            )

            prompt = (
                "You created 3 palettes on canvas. Ask the user to pick one by number or name so the build can continue."
            )
            assistant_msg = await self._generate_friendly_message(prompt)
            return {
                "artboard_id": None,
                "palette_artboard_id": palette_artboard_id,
                "node_ids": [],
                "html_used": "",
                "round": 0,
                "critique": {},
                "done": False,
                "assistant_message": assistant_msg,
                "questions": session.get("questions", []),
                "conversation_stage": "palette_requested",
            }

        if stage == "palette_requested":
            palettes = session.get("palettes") or []
            approved_index = self._extract_approved_palette_index(normalized_brief, palettes)

            if approved_index is None:
                if self._is_palette_approval(normalized_brief):
                    palette_names = [p.get("name", f"Palette {i + 1}") for i, p in enumerate(palettes)]
                    opts = " / ".join(f"'{n}'" for n in palette_names)
                    await self.event_broker.publish(
                        {"type": "palette_clarification_needed", "conversation_id": session_id}
                    )
                    prompt = (
                        f"The user generically approved the palette but didn't say which one. "
                        f"The options are: {opts}. Ask them to clarify which one they want."
                    )
                    assistant_msg = await self._generate_friendly_message(prompt)
                    return {
                        "artboard_id": None,
                        "palette_artboard_id": session.get("palette_artboard_id"),
                        "node_ids": [],
                        "html_used": "",
                        "round": 0,
                        "critique": {},
                        "done": False,
                        "assistant_message": assistant_msg,
                        "questions": session.get("questions", []),
                        "conversation_stage": "palette_requested",
                    }

                palette_names = [p.get("name", f"Palette {i + 1}") for i, p in enumerate(palettes)]
                first_name = palette_names[0] if palette_names else "the first one"
                await self.event_broker.publish(
                    {"type": "palette_approval_required", "conversation_id": session_id, "source": source}
                )
                prompt = (
                    "The user needs to choose a palette to proceed. "
                    f"Ask them to choose one, e.g., 'I like {first_name}'."
                )
                assistant_msg = await self._generate_friendly_message(prompt)
                return {
                    "artboard_id": None,
                    "palette_artboard_id": session.get("palette_artboard_id"),
                    "node_ids": [],
                    "html_used": "",
                    "round": 0,
                    "critique": {},
                    "done": False,
                    "assistant_message": assistant_msg,
                    "questions": session.get("questions", []),
                    "conversation_stage": "palette_requested",
                }

            session["approved_palette"] = palettes[approved_index]
            approved_name = palettes[approved_index].get("name", f"Palette {approved_index + 1}")
            session["palette_summary"] = f"Approved palette: {approved_name}"
            session["stage"] = "content_gathering"
            session["content_questions_pending"] = True

            await self.event_broker.publish(
                {
                    "type": "palette_reviewed",
                    "conversation_id": session_id,
                    "artboard_id": session.get("palette_artboard_id"),
                    "approved_palette": approved_name,
                }
            )

            prompt = (
                "You just approved a palette. Now you need real content from the user "
                "to build their website. "
                "Ask them 3-4 specific questions to gather their product name, features, tagline, and audience. "
                "Ask them in a friendly, conversational way - not as a rigid list."
            )
            assistant_msg = await self._generate_friendly_message(prompt)
            return {
                "artboard_id": None,
                "palette_artboard_id": session.get("palette_artboard_id"),
                "node_ids": [],
                "html_used": "",
                "round": 0,
                "critique": {},
                "done": False,
                "assistant_message": assistant_msg,
                "questions": [],
                "conversation_stage": "content_gathering",
            }

        if stage == "content_gathering":
            if session.get("content_questions_pending", False):
                session["content_questions_pending"] = False

                extract_prompt = (
                    "Extract the website content from the user's response and return strict JSON only:\n"
                    "{\n"
                    '  "name": "project name or null",\n'
                    '  "tagline": "tagline or null",\n'
                    '  "features": "features or null",\n'
                    '  "audience": "target audience or null"\n'
                    "}\n\n"
                    f"User response:\n{normalized_brief}"
                )
                extract_res = await self.designer_llm.ainvoke([HumanMessage(content=extract_prompt)])
                parsed_content = self._try_parse_json(getattr(extract_res, "content", ""))
                if isinstance(parsed_content, dict):
                    session["website_content"] = parsed_content

                session["stage"] = "building"
                await self.event_broker.publish({"type": "content_gathered", "conversation_id": session_id})

        return None

    async def _ask_intake_agent(self, brief: str) -> dict[str, Any]:
        response = await self.designer_llm.ainvoke(
            [
                SystemMessage(content=INTAKE_SYSTEM_PROMPT),
                HumanMessage(content=f"Website brief:\n{brief}"),
            ]
        )
        payload = self._try_parse_json(getattr(response, "content", ""))
        if not isinstance(payload, dict):
            return {
                "genre": "general",
                "questions": [
                    "What is the primary goal of your website?",
                    "Who is the main target audience for this product?",
                ],
            }
        if not isinstance(payload.get("questions"), list) or not payload["questions"]:
            payload["questions"] = [
                "What tone and feeling should the design convey?",
                "What are the most important sections or pages?",
            ]
        if not payload.get("genre"):
            payload["genre"] = "general"
        return payload

    async def _generate_palettes_for_brief(self, brief: str, genre: str) -> list[dict[str, Any]]:
        _fallback_palettes = [
            {
                "name": "Indigo Night",
                "description": "Deep navy and indigo for premium SaaS.",
                "colors": {"bg": "#0a0a0a", "accent": "#6366f1", "surface": "#1a1a1a", "text": "#ffffff", "muted": "#888888", "border": "#2a2a2a"},
            },
            {
                "name": "Amber Signal",
                "description": "Warm amber with dark graphite surfaces.",
                "colors": {"bg": "#111111", "accent": "#f59e0b", "surface": "#1b1b1f", "text": "#ffffff", "muted": "#94a3b8", "border": "#27272a"},
            },
            {
                "name": "Aurora Glass",
                "description": "Cool cyan with soft glass layers.",
                "colors": {"bg": "#07111f", "accent": "#22d3ee", "surface": "#0f172a", "text": "#f1f5f9", "muted": "#64748b", "border": "#1e293b"},
            },
        ]
        try:
            response = await self.designer_llm.ainvoke(
                [
                    SystemMessage(content=PALETTE_SYSTEM_PROMPT),
                    HumanMessage(content=f"Website brief:\n{brief}\n\nGenre: {genre}\n\nGenerate 3 tailored palettes."),
                ]
            )
            payload = self._try_parse_json(getattr(response, "content", ""))
            if isinstance(payload, dict) and isinstance(payload.get("palettes"), list) and payload["palettes"]:
                return payload["palettes"][:3]
        except Exception:
            pass
        return _fallback_palettes

    @staticmethod
    def _is_palette_approval(text: str) -> bool:
        lowered = text.lower()
        approval_tokens = [
            "approve", "approved", "go ahead", "looks good", "proceed", "build it", "start build", "use it",
            "i like", "i love", "let's go", "let's use", "go with", "pick", "choose", "select",
            "palette 1", "palette 2", "palette 3", "option 1", "option 2", "option 3",
            "number 1", "number 2", "number 3", "first one", "second one", "third one",
            "the first", "the second", "the third", "1st", "2nd", "3rd", "#1", "#2", "#3",
            "that one", "this one", "yes", "yeah", "yep", "sure", "perfect", "great", "sounds good", "love it",
        ]
        return any(token in lowered for token in approval_tokens)

    @staticmethod
    def _extract_approved_palette_index(text: str, palettes: list[dict[str, Any]]) -> int | None:
        lowered = text.lower()
        if any(t in lowered for t in ["palette 1", "option 1", "number 1", "the first", "first one", "1st", "#1"]):
            return 0
        if any(t in lowered for t in ["palette 2", "option 2", "number 2", "the second", "second one", "2nd", "#2"]):
            return 1
        if any(t in lowered for t in ["palette 3", "option 3", "number 3", "the third", "third one", "3rd", "#3"]):
            return 2

        for i, palette in enumerate(palettes):
            name = str(palette.get("name", "")).lower().strip()
            if not name:
                continue
            if name in lowered:
                return i
            words = [w for w in name.split() if len(w) > 3]
            if words and words[0] in lowered:
                return i
        return None

    async def _create_palette_artboard(self, brief: str, palettes: list[dict[str, Any]]) -> str:
        if not self.paper_client:
            return ""

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

        palette_html = self._build_palette_html(brief, palettes)
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

    async def _get_paper_tool_names(self) -> set[str]:
        if self._paper_tool_names is not None:
            return self._paper_tool_names
        if not self.paper_client:
            self._paper_tool_names = set()
            return self._paper_tool_names
        try:
            tools = await self.paper_client.list_tools()
            self._paper_tool_names = {
                str(item.get("name", "")).strip()
                for item in tools
                if isinstance(item, dict) and str(item.get("name", "")).strip()
            }
        except Exception:
            self._paper_tool_names = set()
        return self._paper_tool_names

    async def _get_artboard_html_for_refine(self, artboard_id: str, fallback_html: str) -> str:
        """Use in-memory HTML snapshots as refine source-of-truth."""
        return fallback_html

    @staticmethod
    def _build_artboard_html_index(
        artboard_ids: list[str],
        html_used: str,
        markers: tuple[str, ...] = ("<!-- refined-artboard -->", "<!-- artboard-break -->"),
    ) -> dict[str, str]:
        if not artboard_ids:
            return {}

        chunks = [html_used]
        for marker in markers:
            if marker in html_used:
                chunks = [part.strip() for part in html_used.split(marker)]
                break

        if len(chunks) != len(artboard_ids):
            return {}
        return {
            artboard_id: chunk
            for artboard_id, chunk in zip(artboard_ids, chunks)
            if chunk
        }

    async def _call_gemini_flash_refine_hints(
        self,
        *,
        screenshot_b64: str,
        user_feedback: str,
        critique_suggestions: list[str],
        current_html: str,
    ) -> list[str]:
        if not self.gemini_api_key:
            return []

        endpoint = f"{self.gemini_api_base}/models/{self.gemini_refine_model}:generateContent"
        prompt = (
            "You are a UI refinement assistant. Analyze screenshot + current HTML and return strict JSON only.\n"
            "Schema: {\"focused_changes\":[\"...\",\"...\"]}.\n"
            "Rules:\n"
            "- Prioritize explicit user feedback first.\n"
            "- Keep architecture and content intact.\n"
            "- Prefer concrete CSS-level fixes.\n"
            "- Maximum 6 changes.\n"
            f"User feedback: {user_feedback or '(none)'}\n"
            f"Critic suggestions: {json.dumps(critique_suggestions)}\n"
            f"Current HTML excerpt: {current_html[:4000]}"
        )
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": screenshot_b64}},
                    ]
                }
            ]
        }
        headers = {"x-goog-api-key": self.gemini_api_key, "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(endpoint, headers=headers, json=body)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
        if not candidates:
            return []

        parts = candidates[0].get("content", {}).get("parts", []) if isinstance(candidates[0], dict) else []
        text = ""
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text += part["text"]

        parsed = self._try_parse_json(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("focused_changes"), list):
            return [str(item).strip() for item in parsed["focused_changes"] if str(item).strip()][:6]
        return []

    @staticmethod
    def _is_button_radius_request(feedback: str) -> bool:
        text = (feedback or "").lower()
        if not text:
            return False
        has_radius_intent = any(token in text for token in ["radius", "rounded", "round", "corner"])
        has_button_intent = "button" in text or "cta" in text or "try borraaflow today" in text
        return has_radius_intent and has_button_intent

    @staticmethod
    def _extract_radius_style_value(feedback: str) -> str:
        text = (feedback or "").lower()
        m = re.search(r"(\d{1,3})\s*(px)?", text)
        if m:
            return f"{m.group(1)}px"
        if "pill" in text or "fully" in text or "full" in text:
            return "999px"
        return "999px"

    async def _find_button_node_ids_for_feedback(self, artboard_id: str, feedback: str) -> list[str]:
        if not self.paper_client:
            return []

        try:
            result = await self.paper_client.invoke_tool("get_tree_summary", {"nodeId": artboard_id, "depth": 8})
        except Exception:
            return []

        payload = self._try_parse_json(self._extract_primary_text_payload(result))
        if not isinstance(payload, dict):
            return []

        summary_text = str(payload.get("summary", ""))
        if not summary_text:
            return []

        lower_feedback = (feedback or "").lower()
        requested_text = "try borraaflow today" if "borraaflow" in lower_feedback else ""

        text_line_pattern = re.compile(r'Text\s+"[^"]*"\s+\(([^)]+)\)[^\n]*"([^"]+)"')
        matched_text_ids: list[str] = []
        for text_id, text_content in text_line_pattern.findall(summary_text):
            content = str(text_content).strip().lower()
            if requested_text:
                if requested_text in content:
                    matched_text_ids.append(str(text_id).strip())
            elif "button" in lower_feedback or "cta" in lower_feedback:
                matched_text_ids.append(str(text_id).strip())

        parent_button_ids: list[str] = []
        for text_id in matched_text_ids:
            try:
                info_result = await self.paper_client.invoke_tool("get_node_info", {"nodeId": text_id})
                info = self._try_parse_json(self._extract_primary_text_payload(info_result))
                if isinstance(info, dict):
                    parent_id = info.get("parentId")
                    if isinstance(parent_id, str) and parent_id.strip():
                        parent_button_ids.append(parent_id.strip())
            except Exception:
                continue

        if parent_button_ids:
            return list(dict.fromkeys(parent_button_ids))

        frame_button_ids = re.findall(r'Frame\s+"Button"\s+\(([^)]+)\)', summary_text)
        return list(dict.fromkeys([str(node_id).strip() for node_id in frame_button_ids if str(node_id).strip()]))

    async def _apply_targeted_style_patch(self, artboard_id: str, feedback: str) -> bool:
        if not self.paper_client or not self._is_button_radius_request(feedback):
            return False

        button_node_ids = await self._find_button_node_ids_for_feedback(artboard_id, feedback)
        if not button_node_ids:
            print("[Pipeline][refine] update_styles skipped: no matching button nodes found")
            return False

        radius_value = self._extract_radius_style_value(feedback)
        payload = {
            "updates": [
                {
                    "nodeIds": button_node_ids,
                    "styles": {
                        "borderRadius": radius_value,
                    },
                }
            ]
        }

        try:
            result = await self.paper_client.invoke_tool("update_styles", payload)
            self._log_paper_result(
                "refine",
                "update_styles",
                result,
                detail=f"patched_nodes={len(button_node_ids)} borderRadius={radius_value}",
            )
            return not bool(result.get("isError"))
        except Exception as exc:
            print(f"[Pipeline][refine] update_styles failed: {exc}")
            return False

    @staticmethod
    def _paper_error_message(result: dict[str, Any]) -> str:
        text = VibeframeAgentPipeline._extract_primary_text_payload(result).strip()
        return text or "Unknown MCP error"

    def _log_paper_result(self, stage: str, tool_name: str, result: dict[str, Any], *, detail: str = "") -> None:
        prefix = f"[Paper MCP][{stage}][{tool_name}]"
        if result.get("isError"):
            print(f"{prefix} ERROR: {self._paper_error_message(result)}")
        else:
            suffix = f" {detail}" if detail else ""
            print(f"{prefix} OK{suffix}")

    @staticmethod
    def _enforce_full_bleed_html(html: str) -> str:
        # Make the root container fill the artboard and remove outer whitespace.
        def _rewrite(match: re.Match[str]) -> str:
            style = match.group(1)
            style = re.sub(r"(?i)max-width\s*:\s*[^;]+;?", "", style)
            style = re.sub(r"(?i)margin\s*:\s*[^;]+;?", "", style)
            style = re.sub(r"(?i)padding\s*:\s*[^;]+;?", "", style)
            style = style.strip().rstrip(";")
            enforced = "width:100%;min-height:100%;margin:0;padding:0;box-sizing:border-box"
            merged = f"{style};{enforced}" if style else enforced
            return f"<div style='{merged}'>"

        return re.sub(r"<div\s+style=['\"]([^'\"]*)['\"]>", _rewrite, html, count=1, flags=re.IGNORECASE)

    @staticmethod
    def _is_invalid_html_candidate(html: str) -> bool:
        text = html.strip().lower()
        if not text:
            return True

        error_markers = [
            "unknown tool:",
            "html produced no design nodes",
            "invalid arguments for tool",
            "paper mcp error",
        ]
        if any(marker in text for marker in error_markers):
            return True

        return False

    @traceable
    async def _generate_friendly_message(self, prompt: str) -> str:
        system_content = """You are Vibeframe, a friendly and enthusiastic AI web designer. 
You speak like a talented creative colleague — warm, excited about the work, occasionally playful. Never robotic. 
Never use bullet points or bold markdown in your spoken responses. Keep it conversational, like texting a friend who happens to be a great designer. 2-3 sentences max."""
        try:
            response = await self.designer_llm.ainvoke(
                [
                    SystemMessage(content=system_content),
                    HumanMessage(content=prompt),
                ]
            )
            return str(getattr(response, "content", "Let's get started!")).strip()
        except Exception:
            return "Let's get started!"

    # --- Node Implementations ---
    @traceable
    async def intake_node(self, state: PipelineState) -> PipelineState:
        """Analyze brief and detect genre + emotional direction."""
        print("LOG: Running Intake Node...")
        response = await self.designer_llm.ainvoke(
            [
                SystemMessage(content=INTAKE_SYSTEM_PROMPT),
                HumanMessage(content=f"Website Brief: \n{state.get('brief')}")
            ]
        )
        
        payload = self._try_parse_json(getattr(response, "content", ""))
        
        if not isinstance(payload, dict):
            payload = {
                "genre": "general",
                "emotional_direction": "professional",
                "questions": [
                    "What is the primary goal of your website?",
                    "Who is the main target audience for this product?",
                ],
            }
        
        genre = payload.get("genre", "general")
        emotional_direction = payload.get("emotional_direction", "professional")
        questions = payload.get("questions", [])
        if not isinstance(questions, list) or not questions:
            questions = ["What tone and feeling should the design convey?", "What are the most important sections?"]
        
        await self.event_broker.publish({
            "type": "intake_completed",
            "genre": genre,
            "emotional_direction": emotional_direction
        })
        
        return {
            **state,
            "genre": genre,
            "emotional_direction": emotional_direction,
            "intake_questions": questions,
            "stage": "intake_confirm"
        }

    @traceable
    async def intake_confirm_node(self, state: PipelineState) -> PipelineState:
        """INTERRUPT: Show intake questions and wait for user answers."""
        print("LOG: Running Intake Confirm Node (INTERRUPT)...")
        
        questions = state.get("intake_questions", [])
        genre = state.get("genre", "general")
        
        prompt = f"The user wants to build a {genre} website. Ask them these clarifying questions naturally and conversationally."
        friendly_msg = await self._generate_friendly_message(prompt)
        
        user_response = interrupt({
            "type": "intake_confirmation",
            "message": friendly_msg,
            "questions": questions,
            "genre": genre
        })
        
        await self.event_broker.publish({
            "type": "intake_confirmed",
            "genre": genre,
            "questions_count": len(questions)
        })
        
        return {
            **state,
            "intake_answers": user_response.get("answers", {}),
            "stage": "palette"
        }

    @traceable
    async def palette_node(self, state: PipelineState) -> PipelineState:
        """Generate 3 tailored palettes and create Paper artboard."""
        print("LOG: Running Palette Node...")
        
        _fallback_palettes = [
            {
                "name": "Indigo Night",
                "personality": "expected",
                "description": "Deep navy and indigo for premium SaaS.",
                "typography_weight": "balanced",
                "colors": {"bg": "#0a0a0a", "accent": "#6366f1", "surface": "#1a1a1a", "text": "#ffffff", "muted": "#888888", "border": "#2a2a2a"},
            },
            {
                "name": "Amber Signal",
                "personality": "bold",
                "description": "Warm amber with dark graphite surfaces.",
                "typography_weight": "heavy",
                "colors": {"bg": "#111111", "accent": "#f59e0b", "surface": "#1b1b1f", "text": "#ffffff", "muted": "#94a3b8", "border": "#27272a"},
            },
            {
                "name": "Aurora Glass",
                "personality": "refined",
                "description": "Cool cyan with soft glass layers.",
                "typography_weight": "light",
                "colors": {"bg": "#07111f", "accent": "#22d3ee", "surface": "#0f172a", "text": "#f1f5f9", "muted": "#64748b", "border": "#1e293b"},
            },
        ]
        
        brief = state.get("brief", "")
        genre = state.get("genre", "general")
        emotional_direction = state.get("emotional_direction", "professional")
        
        try:
            response = await self.designer_llm.ainvoke(
                [
                    SystemMessage(content=PALETTE_SYSTEM_PROMPT),
                    HumanMessage(content=f"Website brief:\n{brief}\n\nGenre: {genre}\n\nEmotional direction: {emotional_direction}\n\nGenerate 3 tailored palettes."),
                ]
            )
            payload = self._try_parse_json(getattr(response, "content", ""))
            if isinstance(payload, dict) and isinstance(payload.get("palettes"), list) and payload["palettes"]:
                palettes = payload["palettes"][:3]
            else:
                palettes = _fallback_palettes
        except Exception as e:
            print(f"Palette generation error: {e}")
            palettes = _fallback_palettes
        
        # Create Paper artboard for palettes (optional if paper_client is available)
        palette_id = None
        if self.paper_client:
            try:
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
                if palette_id:
                    palette_html = self._build_palette_html(brief, palettes)
                    write_result = await self.paper_client.invoke_tool(
                        "write_html",
                        {
                            "html": palette_html,
                            "targetNodeId": palette_id,
                            "mode": "insert-children",
                        },
                    )
                    if not write_result.get("isError"):
                        await self.event_broker.publish({
                            "type": "palette_artboard_created",
                            "artboard_id": palette_id,
                            "palette_count": len(palettes)
                        })
            except Exception as e:
                print(f"Paper artboard error: {e}")
                palette_id = None
        
        return {
            **state,
            "palettes": palettes,
            "palette_artboard_id": palette_id,
            "stage": "palette_confirm"
        }

    @traceable
    async def palette_confirm_node(self, state: PipelineState) -> PipelineState:
        """INTERRUPT: Show palettes and wait for user to pick one."""
        print("LOG: Running Palette Confirm Node (INTERRUPT)...")
        
        palettes = state.get("palettes", [])
        palette_names = [p.get("name", f"Palette {i+1}") for i, p in enumerate(palettes)]
        
        prompt = f"You've created 3 unique palette directions. Ask the user which one they prefer: {', '.join(palette_names)}."
        friendly_msg = await self._generate_friendly_message(prompt)
        
        user_response = interrupt({
            "type": "palette_selection",
            "message": friendly_msg,
            "palette_names": palette_names,
            "palette_artboard_id": state.get("palette_artboard_id"),
        })
        
        # Extract palette index from response
        selected_index = user_response.get("selected_index", 0)
        if not isinstance(selected_index, int) or selected_index >= len(palettes):
            selected_index = 0
        
        selected_palette = palettes[selected_index]
        
        await self.event_broker.publish({
            "type": "palette_selected",
            "palette_name": selected_palette.get("name"),
            "index": selected_index
        })
        
        return {
            **state,
            "approved_palette": selected_palette,
            "stage": "content_gather"
        }

    @traceable
    async def content_gather_node(self, state: PipelineState) -> PipelineState:
        """Generate content gathering questions based on genre."""
        print("LOG: Running Content Gather Node...")
        
        brief = state.get("brief", "")
        genre = state.get("genre", "general")
        
        # Generate content-specific questions
        prompt = f"""Based on this {genre} website brief, generate 3-4 specific questions to gather:
- Product name or business name
- Tagline or one-liner
- Key features or services (2-3)
- Target audience

Brief: {brief}

Return as JSON: {{"questions": ["q1", "q2", "q3", "q4"]}}"""
        
        try:
            response = await self.designer_llm.ainvoke(
                [
                    SystemMessage(content="You are a UX researcher. Generate specific content-gathering questions."),
                    HumanMessage(content=prompt),
                ]
            )
            payload = self._try_parse_json(getattr(response, "content", ""))
            questions = payload.get("questions", []) if isinstance(payload, dict) else []
        except Exception:
            questions = []
        
        if not questions:
            questions = [
                "What's the name of your product or business?",
                "Can you give me a one-liner tagline?",
                "What are your 3-4 key features or services?",
                "Who is your ideal audience?",
            ]
        
        await self.event_broker.publish({
            "type": "content_questions_generated",
            "question_count": len(questions)
        })
        
        return {
            **state,
            "content_questions": questions,
            "stage": "content_confirm"
        }

    @traceable
    async def content_confirm_node(self, state: PipelineState) -> PipelineState:
        """INTERRUPT: Show content questions and collect user answers."""
        print("LOG: Running Content Confirm Node (INTERRUPT)...")
        
        questions = state.get("content_questions", [])
        approved_palette = state.get("approved_palette", {})
        
        prompt = "Now let's gather the content for your website. Ask the user the content questions in a natural, conversational way."
        friendly_msg = await self._generate_friendly_message(prompt)
        
        user_response = interrupt({
            "type": "content_confirmation",
            "message": friendly_msg,
            "questions": questions,
            "palette_name": (approved_palette.get("name", "Selected Palette") if approved_palette else "Selected Palette"),
        })
        
        answers = user_response.get("answers", {})
        
        # Parse content from answers
        website_content = {
            "name": answers.get("name", "Your Product"),
            "tagline": answers.get("tagline", ""),
            "features": answers.get("features", ""),
            "audience": answers.get("audience", ""),
        }
        
        await self.event_broker.publish({
            "type": "content_collected",
            "product_name": website_content.get("name"),
        })
        
        return {
            **state,
            "content_answers": answers,
            "website_content": website_content,
            "stage": "section_planning"
        }

    @traceable
    async def generate_adaptive_question(
        self,
        *,
        stage: Literal["intake", "content"],
        brief: str,
        genre: str,
        emotional_direction: str,
        asked_questions: list[str],
        collected_answers: dict[str, str],
        max_questions: int,
        prior_answers: Optional[dict[str, str]] = None,
        prior_questions: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Generate the next question after evaluating all previously collected answers."""
        if len(collected_answers) >= max_questions:
            return {"done": True, "question": "", "key": ""}

        prior_answers = prior_answers or {}
        prior_questions = prior_questions or []

        stage_focus = (
            "For intake, discover audience, tone, value proposition, and key sections."
            if stage == "intake"
            else "For content, collect product name, tagline, features/services, audience, and CTA intent."
        )

        allowed_keys = (
            "audience, emotion, value_prop, differentiation, key_sections, goal"
            if stage == "intake"
            else "name, tagline, features, audience, cta"
        )

        prompt = (
            "You are generating one next best question for a website discovery interview. "
            "Use the brief and previous answers to ask a smart follow-up question.\n\n"
            f"Stage: {stage}\n"
            f"Brief: {brief}\n"
            f"Genre: {genre}\n"
            f"Emotional direction: {emotional_direction}\n"
            f"Prior-stage questions (already asked): {json.dumps(prior_questions)}\n"
            f"Prior-stage answers (already collected): {json.dumps(prior_answers)}\n"
            f"Asked questions so far: {json.dumps(asked_questions)}\n"
            f"Collected answers so far: {json.dumps(collected_answers)}\n\n"
            f"{stage_focus}\n"
            f"Allowed answer keys: {allowed_keys}\n\n"
            "Return strict JSON only with this exact schema:\n"
            '{"done": false, "question": "...", "key": "..."}\n\n'
            "Rules:\n"
            "- Ask only one question.\n"
            "- No repeated or near-duplicate questions.\n"
            "- Do not ask for information already answered in prior-stage answers.\n"
            "- Question must be specific to this brief and previous answers.\n"
            "- Keep it concise and conversational.\n"
            "- If enough information has been collected, return done=true and empty question/key."
        )

        try:
            response = await self.designer_llm.ainvoke(
                [
                    SystemMessage(content="You are a senior UX interviewer. Always return strict JSON only."),
                    HumanMessage(content=prompt),
                ]
            )
            payload = self._try_parse_json(getattr(response, "content", ""))
            if isinstance(payload, dict):
                done = bool(payload.get("done", False))
                question = str(payload.get("question", "")).strip()
                key = str(payload.get("key", "")).strip().lower()
                if done:
                    return {"done": True, "question": "", "key": ""}
                if question:
                    return {"done": False, "question": question, "key": key or f"{stage}_{len(collected_answers)+1}"}
        except Exception:
            pass

        # Fallback sequence in case the model output is malformed.
        if stage == "intake":
            fallback = [
                ("Who is the primary audience, and what should they feel in the first 5 seconds?", "audience"),
                ("What outcome should visitors achieve immediately on the landing page?", "goal"),
                ("Which sections are mandatory in v1: features, pricing, testimonials, FAQ, or contact?", "key_sections"),
            ]
        else:
            fallback = [
                ("What exact product or business name should appear in the hero?", "name"),
                ("What one-line tagline should sit under the headline?", "tagline"),
                ("List 3 core features or services (comma-separated).", "features"),
                ("Who is the ideal audience we should speak to directly?", "audience"),
            ]

        idx = len(collected_answers)
        if idx >= len(fallback):
            return {"done": True, "question": "", "key": ""}
        question, key = fallback[idx]
        return {"done": False, "question": question, "key": key}

    @staticmethod
    def _normalize_answer_map(*answer_sets: Optional[dict[str, str]]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for answer_set in answer_sets:
            if not answer_set:
                continue
            for raw_key, raw_value in answer_set.items():
                key = str(raw_key).strip().lower().replace("-", "_").replace(" ", "_")
                value = str(raw_value).strip()
                if key and value:
                    normalized[key] = value
        return normalized

    @staticmethod
    def _pick_first_non_empty(answers: dict[str, str], keys: list[str]) -> str:
        for key in keys:
            value = answers.get(key, "")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def build_website_content_from_answers(
        answers: dict[str, str],
        *,
        intake_answers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Normalize collected answers into rich website content payload.

        Content answers take priority, with intake answers as fallback context.
        """
        merged = VibeframeAgentPipeline._normalize_answer_map(intake_answers, answers)

        name = VibeframeAgentPipeline._pick_first_non_empty(
            merged,
            ["name", "product_name", "brand_name", "business_name", "company_name", "content_1"],
        ) or "Your Product"
        tagline = VibeframeAgentPipeline._pick_first_non_empty(
            merged,
            ["tagline", "one_liner", "headline", "value_prop", "content_2"],
        )
        features_raw = VibeframeAgentPipeline._pick_first_non_empty(
            merged,
            ["features", "services", "capabilities", "feature_list", "content_3", "key_sections", "differentiation"],
        )
        audience = VibeframeAgentPipeline._pick_first_non_empty(
            merged,
            ["audience", "target_audience", "users", "customer", "persona", "content_4"],
        )
        goal = VibeframeAgentPipeline._pick_first_non_empty(
            merged,
            ["goal", "primary_goal", "cta", "outcome"],
        )

        features: list[str]
        if isinstance(features_raw, str) and features_raw.strip():
            raw_parts = (
                features_raw.replace(";", ",")
                .replace("\n", ",")
                .replace("|", ",")
                .split(",")
            )
            features = [part.strip(" -•\t") for part in raw_parts if part.strip(" -•\t")]
        else:
            features = []

        if not features and goal:
            features = [goal]

        return {
            "name": name,
            "tagline": tagline,
            "features": features,
            "audience": audience,
            "goal": goal,
            "sample_copy": {},
        }

    @traceable
    async def enrich_website_content(
        self,
        *,
        brief: str,
        genre: str,
        emotional_direction: str,
        intake_answers: dict[str, str],
        content_answers: dict[str, str],
        website_content: dict[str, Any],
    ) -> dict[str, Any]:
        """Fill missing content fields and generate reusable sample copy for future design nodes."""
        prompt = (
            "You are a senior website content strategist. Build high-quality website content JSON from inputs.\n\n"
            f"Brief: {brief}\n"
            f"Genre: {genre}\n"
            f"Emotional direction: {emotional_direction}\n"
            f"Intake answers: {json.dumps(intake_answers)}\n"
            f"Content answers: {json.dumps(content_answers)}\n"
            f"Current website content: {json.dumps(website_content)}\n\n"
            "Return strict JSON only with this schema:\n"
            "{"
            '"name":"...",'
            '"tagline":"...",'
            '"audience":"...",'
            '"goal":"...",'
            '"features":["...","...","..."],'
            '"sample_copy":{'
            '"hero_headline":"...",'
            '"hero_subheadline":"...",'
            '"primary_cta":"...",'
            '"secondary_cta":"...",'
            '"feature_cards":["...","...","..."],'
            '"testimonial_quote":"...",'
            '"faq_items":["...","...","..."]'
            "}"
            "}\n\n"
            "Rules:\n"
            "- Prefer user-provided facts over invented content.\n"
            "- If a field is missing, infer plausible and concise copy from the brief.\n"
            "- features must be 3-6 short bullets.\n"
            "- Keep tone aligned with emotional direction.\n"
            "- Do not include markdown fences."
        )

        try:
            response = await self.designer_llm.ainvoke(
                [
                    SystemMessage(content="You produce strict JSON website copy only."),
                    HumanMessage(content=prompt),
                ]
            )
            payload = self._try_parse_json(getattr(response, "content", ""))
            if isinstance(payload, dict):
                enriched = {**website_content}
                for field in ["name", "tagline", "audience", "goal"]:
                    value = payload.get(field)
                    if isinstance(value, str) and value.strip():
                        enriched[field] = value.strip()

                features = payload.get("features")
                if isinstance(features, list):
                    clean_features = [str(item).strip() for item in features if str(item).strip()]
                    if clean_features:
                        enriched["features"] = clean_features[:6]

                sample_copy = payload.get("sample_copy")
                if isinstance(sample_copy, dict):
                    enriched["sample_copy"] = sample_copy

                return enriched
        except Exception:
            pass

        return website_content

    @staticmethod
    def summarize_content_completeness(
        website_content: dict[str, Any],
        *,
        intake_answers: Optional[dict[str, str]] = None,
        content_answers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        merged = VibeframeAgentPipeline._normalize_answer_map(intake_answers, content_answers)

        def _is_direct(value: Any) -> bool:
            if not isinstance(value, str) or not value.strip():
                return False
            haystack = " ".join(merged.values()).lower()
            needle = value.strip().lower()
            return needle in haystack

        features_value = website_content.get("features", [])
        features_list = features_value if isinstance(features_value, list) else []
        features_direct = any(_is_direct(item) for item in features_list)

        field_sources = {
            "name": "provided" if _is_direct(website_content.get("name", "")) else "inferred/default",
            "tagline": "provided" if _is_direct(website_content.get("tagline", "")) else "inferred/default",
            "audience": "provided" if _is_direct(website_content.get("audience", "")) else "inferred/default",
            "features": "provided" if features_direct else "inferred/default",
        }

        missing_fields = [
            field
            for field in ["name", "tagline", "audience"]
            if not str(website_content.get(field, "")).strip()
        ]

        return {
            "field_sources": field_sources,
            "missing_fields": missing_fields,
            "feature_count": len(features_list),
            "has_sample_copy": bool(website_content.get("sample_copy")),
        }

    @staticmethod
    def _extract_created_node_ids(result: dict[str, Any]) -> list[str]:
        parsed = VibeframeAgentPipeline._try_parse_json(
            VibeframeAgentPipeline._extract_primary_text_payload(result)
        )
        node_ids: list[str] = []
        if isinstance(parsed, dict):
            for key in ["nodeIds", "node_ids", "ids"]:
                value = parsed.get(key)
                if isinstance(value, list):
                    node_ids.extend([str(v) for v in value if str(v).strip()])
            for key in ["id", "nodeId", "node_id"]:
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    node_ids.append(value)
        return list(dict.fromkeys(node_ids))

    @staticmethod
    def _extract_image_base64_payload(result: dict[str, Any]) -> Optional[str]:
        image_payload = VibeframeAgentPipeline._extract_image_payload(result)
        if image_payload:
            return image_payload[1]
        return None

    @staticmethod
    def _extract_image_payload(value: Any) -> tuple[str, str] | None:
        if isinstance(value, dict):
            for key in ("dataUrl", "data_url", "imageDataUrl", "image_data_url"):
                data_url = value.get(key)
                if isinstance(data_url, str):
                    parsed = VibeframeAgentPipeline._parse_data_url(data_url)
                    if parsed:
                        return parsed

            for key in ("base64", "imageBase64", "image_base64", "pngBase64", "data"):
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

            parsed_json = VibeframeAgentPipeline._try_parse_json(value)
            if isinstance(parsed_json, (dict, list)):
                return VibeframeAgentPipeline._extract_image_payload(parsed_json)

        return None

    @staticmethod
    def _parse_data_url(value: str) -> tuple[str, str] | None:
        if not value.startswith("data:"):
            return None
        if ";base64," not in value:
            return None

        header, payload = value.split(",", 1)
        mime_type = header[5:].split(";", 1)[0].strip() or "image/png"
        payload = payload.strip()
        if not payload:
            return None
        return mime_type, payload

    @traceable
    async def section_planner_node(self, state: PipelineState) -> PipelineState:
        brief = str(state.get("brief", ""))
        genre = str(state.get("genre", "general") or "general")
        emotional_direction = str(state.get("emotional_direction", "professional") or "professional")
        website_content = state.get("website_content", {}) or {}
        approved_palette = state.get("approved_palette", {}) or {}

        planner_input = (
            f"Brief: {brief}\n"
            f"Genre: {genre}\n"
            f"Emotional direction: {emotional_direction}\n"
            f"Website content: {json.dumps(website_content)}\n"
            f"Approved palette: {json.dumps(approved_palette)}"
        )

        fallback_plan = {
            "multi_artboard": False,
            "design_rationale": "Single-page flow keeps launch focused and fast for first release.",
            "artboards": [
                {
                    "name": "Home",
                    "width": 1440,
                    "sections": ["nav", "hero", "features", "how_it_works", "cta", "footer"],
                    "layout_style": "centered",
                    "hero_style": "centered",
                }
            ],
        }

        page_plan: dict[str, Any] = fallback_plan
        try:
            response = await self.designer_llm.ainvoke(
                [
                    SystemMessage(content=SECTION_PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=planner_input),
                ]
            )
            payload = self._try_parse_json(getattr(response, "content", ""))
            if isinstance(payload, dict) and isinstance(payload.get("artboards"), list) and payload["artboards"]:
                artboards = payload["artboards"][:4]
                normalized_artboards: list[dict[str, Any]] = []
                for idx, art in enumerate(artboards):
                    if not isinstance(art, dict):
                        continue
                    name = str(art.get("name", f"Page {idx+1}"))
                    width = int(art.get("width", 1440)) if str(art.get("width", "")).strip().isdigit() else 1440
                    sections = art.get("sections", [])
                    if not isinstance(sections, list) or not sections:
                        sections = ["nav", "hero", "features", "cta", "footer"]
                    layout_style = str(art.get("layout_style", "centered"))
                    hero_style = str(art.get("hero_style", "centered"))
                    normalized_artboards.append(
                        {
                            "name": name,
                            "width": width,
                            "sections": [str(s) for s in sections],
                            "layout_style": layout_style,
                            "hero_style": hero_style,
                        }
                    )
                page_plan = {
                    "multi_artboard": bool(payload.get("multi_artboard", len(normalized_artboards) > 1)),
                    "design_rationale": str(payload.get("design_rationale", "")) or fallback_plan["design_rationale"],
                    "artboards": normalized_artboards or fallback_plan["artboards"],
                }
        except Exception:
            page_plan = fallback_plan

        await self.event_broker.publish({"type": "section_plan_ready", "plan": page_plan})
        return {**state, "page_plan": page_plan, "stage": "designer"}

    @traceable
    async def designer_node(self, state: PipelineState) -> PipelineState:
        page_plan = state.get("page_plan", {}) or {}
        artboards = page_plan.get("artboards", []) if isinstance(page_plan, dict) else []
        existing_artboard_ids = [
            str(item)
            for item in (state.get("artboard_ids", []) or [])
            if isinstance(item, str) and item.strip()
        ]
        website_content = state.get("website_content", {}) or {}
        palette = state.get("approved_palette", {}) or {}
        colors = palette.get("colors", {}) if isinstance(palette, dict) else {}

        bg = str(colors.get("bg", "#0a0a0a"))
        bg_alt = str(colors.get("bg_alt", "#111111"))
        surface = str(colors.get("surface", "#1a1a1a"))
        accent = str(colors.get("accent", "#6366f1"))
        accent_muted = str(colors.get("accent_muted", "#4f46e533"))
        text = str(colors.get("text", "#ffffff"))
        text_muted = str(colors.get("text_muted", "#a1a1aa"))
        border = str(colors.get("border", "#2a2a2a"))
        gradient_start = str(colors.get("gradient_start", bg))
        gradient_end = str(colors.get("gradient_end", bg_alt))
        button_text = str(colors.get("button_text", "#ffffff"))
        typography_weight = str(palette.get("typography_weight", "balanced"))

        product_name = str(website_content.get("name", "Your Product"))
        tagline = str(website_content.get("tagline", ""))
        features = website_content.get("features", [])
        features_text = ", ".join([str(x) for x in features]) if isinstance(features, list) else str(features)
        audience = str(website_content.get("audience", ""))

        artboard_ids: list[str] = []
        all_node_ids: list[str] = []
        html_snapshots: list[str] = []
        last_html_by_artboard: dict[str, str] = {}

        if not isinstance(artboards, list) or not artboards:
            artboards = [
                {
                    "name": "Home",
                    "width": 1440,
                    "sections": ["nav", "hero", "features", "cta", "footer"],
                    "layout_style": "centered",
                    "hero_style": "centered",
                }
            ]

        for idx, artboard in enumerate(artboards):
            art = artboard if isinstance(artboard, dict) else {}
            artboard_name = str(art.get("name", f"Page {idx+1}"))
            sections = art.get("sections", []) if isinstance(art.get("sections", []), list) else []
            layout_style = str(art.get("layout_style", "centered"))
            hero_style = str(art.get("hero_style", "centered"))

            artboard_id: Optional[str] = None
            if self.paper_client:
                if idx < len(existing_artboard_ids):
                    artboard_id = existing_artboard_ids[idx]
                    print("\n" + "="*80)
                    print(f"♻️  REUSING ARTBOARD: {artboard_name} ({artboard_id})")
                    print("="*80)
                    try:
                        await self._clear_artboard(artboard_id)
                    except Exception as exc:
                        print(f"[Paper MCP] Could not clear artboard {artboard_id}: {exc}")
                else:
                    create_payload = {
                        "name": artboard_name,
                        "styles": {
                            "width": "1440px",
                            "height": "900px",
                            "minHeight": "900px",
                            "backgroundColor": bg,
                            "display": "flex",
                            "flexDirection": "column",
                        },
                    }
                    print(f"[Pipeline][designer] create_artboard requested: name={artboard_name}")

                    create_result = await self.paper_client.invoke_tool(
                        "create_artboard",
                        create_payload,
                    )
                    self._log_paper_result("designer", "create_artboard", create_result)

                    artboard_id = self._extract_node_id_from_create_artboard(create_result)

            user_message = (
                f"Build the '{artboard_name}' page for {product_name}.\n\n"
                f"Layout style: {layout_style}\n"
                f"Hero style: {hero_style}\n"
                f"Sections to include IN ORDER: {sections}\n"
                f"Emotional direction: {state.get('emotional_direction', 'professional')}\n\n"
                f"Design system (use these exact values):\n"
                f"Background: {bg}\n"
                f"Alt background: {bg_alt}\n"
                f"Surface/cards: {surface}\n"
                f"Accent: {accent}\n"
                f"Accent muted: {accent_muted}\n"
                f"Text: {text}\n"
                f"Text muted: {text_muted}\n"
                f"Border: {border}\n"
                f"Gradient: {gradient_start} -> {gradient_end}\n"
                f"Button text: {button_text}\n"
                f"Typography weight: {typography_weight}\n\n"
                f"Real content to use (never use placeholders):\n"
                f"Product name: {product_name}\n"
                f"Tagline: {tagline}\n"
                f"Features: {features_text}\n"
                f"Audience: {audience}\n\n"
                'Return strict JSON: {"summary": "...", "html": "..."}'
            )

            generated_html = ""
            try:
                response = await self.designer_llm.ainvoke(
                    [
                        SystemMessage(content=DESIGNER_SYSTEM_PROMPT),
                        HumanMessage(content=user_message),
                    ]
                )
                payload = self._try_parse_json(getattr(response, "content", ""))
                if isinstance(payload, dict):
                    generated_html = str(payload.get("html", "")).strip()
            except Exception:
                generated_html = ""

            if not generated_html:
                generated_html = (
                    f"<div style='min-height:100vh;background:{bg};color:{text};padding:48px;font-family:Inter,system-ui,sans-serif;'>"
                    f"<h1 style='font-size:56px;margin:0 0 12px 0;color:{text};'>{product_name}</h1>"
                    f"<p style='font-size:20px;color:{text_muted};margin:0 0 24px 0;'>{tagline or 'Built for modern teams.'}</p>"
                    f"<div style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;'>"
                    + "".join(
                        [
                            f"<div style='background:{surface};border:1px solid {border};padding:16px;border-radius:16px;'>{f}</div>"
                            for f in (features if isinstance(features, list) and features else ["Feature 1", "Feature 2", "Feature 3"])
                        ]
                    )
                    + "</div></div>"
                )

            generated_html = self._enforce_full_bleed_html(generated_html)

            html_snapshots.append(generated_html)

            if self.paper_client and artboard_id:
                write_payload = {
                    "html": generated_html,
                    "targetNodeId": artboard_id,
                    "mode": "insert-children",
                }
                print(
                    f"[Pipeline][designer] write_html requested: artboard={artboard_id}, html_chars={len(generated_html)}"
                )
                
                write_result = await self.paper_client.invoke_tool(
                    "write_html",
                    write_payload,
                )
                created_count = len(self._extract_created_node_ids(write_result))
                self._log_paper_result(
                    "designer",
                    "write_html",
                    write_result,
                    detail=f"created_nodes={created_count}",
                )
                
                all_node_ids.extend(self._extract_created_node_ids(write_result))
                artboard_ids.append(artboard_id)
                last_html_by_artboard[artboard_id] = generated_html

                await self.event_broker.publish(
                    {"type": "artboard_built", "name": artboard_name, "artboard_id": artboard_id}
                )

        html_used = "\n\n<!-- artboard-break -->\n\n".join(html_snapshots)
        return {
            **state,
            "artboard_ids": artboard_ids,
            "last_html_by_artboard": last_html_by_artboard or state.get("last_html_by_artboard", {}),
            "node_ids": list(dict.fromkeys(all_node_ids)),
            "html_used": html_used,
            "current_html": html_snapshots[-1] if html_snapshots else state.get("current_html", ""),
            "stage": "critic",
        }

    async def _call_gemini_vision_critic(self, screenshot_b64: str, context_text: str) -> Optional[dict[str, Any]]:
        if not self.gemini_api_key:
            return None

        endpoint = f"{self.gemini_api_base}/models/{self.gemini_critic_model}:generateContent"
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{CRITIC_SYSTEM_PROMPT}\n\nContext:\n{context_text}"},
                        {"inline_data": {"mime_type": "image/png", "data": screenshot_b64}},
                    ]
                }
            ]
        }
        headers = {"x-goog-api-key": self.gemini_api_key, "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()

        candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", []) if isinstance(candidates[0], dict) else []
        text = ""
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text += part["text"]
        parsed = self._try_parse_json(text)
        return parsed if isinstance(parsed, dict) else None

    async def _fallback_html_critic(self, html_text: str, context_text: str) -> dict[str, Any]:
        try:
            response = await self.critic_llm.ainvoke(
                [
                    SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                    HumanMessage(content=f"Context:\n{context_text}\n\nHTML:\n{html_text}"),
                ]
            )
            payload = self._try_parse_json(getattr(response, "content", ""))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {
            "score": 6,
            "issues": ["Critic fallback used due to unavailable vision analysis."],
            "suggestions": ["Improve hierarchy, spacing, and visual contrast."],
        }

    @traceable
    async def critic_node(self, state: PipelineState) -> PipelineState:
        artboard_ids = state.get("artboard_ids", []) or []
        target_score = int(state.get("target_score", 8) or 8)
        current_round = int(state.get("round", 0) or 0)
        max_rounds = int(state.get("max_rounds", 3) or 3)

        all_scores: list[int] = []
        all_issues: list[str] = []
        all_suggestions: list[str] = []

        context_text = (
            f"Brief: {state.get('brief', '')}\n"
            f"Website content: {json.dumps(state.get('website_content', {}))}"
        )

        tool_names = await self._get_paper_tool_names() if self.paper_client else set()
        has_screenshot_tool = "get_screenshot" in tool_names

        for artboard_id in artboard_ids:
            critique_payload: dict[str, Any]
            gemini_used = False
            if self.paper_client:
                try:
                    screenshot_b64: Optional[str] = None
                    if has_screenshot_tool:
                        screenshot_result = await self.paper_client.invoke_tool(
                            "get_screenshot", {"nodeId": artboard_id}
                        )
                        screenshot_b64 = self._extract_image_base64_payload(screenshot_result)

                    gemini_available = time.time() >= self._gemini_circuit_open_until
                    if screenshot_b64 and gemini_available:
                        try:
                            vision_result = await self._call_gemini_vision_critic(screenshot_b64, context_text)
                            if isinstance(vision_result, dict):
                                critique_payload = vision_result
                                gemini_used = True
                            else:
                                raise ValueError("Gemini returned empty critic payload")
                        except Exception:
                            self._gemini_circuit_open_until = time.time() + 120.0
                            critique_payload = await self._fallback_html_critic(
                                str(state.get("html_used", "")), context_text
                            )
                    else:
                        if not has_screenshot_tool:
                            print("[Paper MCP] get_screenshot unavailable; critic will use HTML fallback.")
                        critique_payload = await self._fallback_html_critic(
                            str(state.get("html_used", "")), context_text
                        )
                except Exception:
                    critique_payload = await self._fallback_html_critic(
                        str(state.get("html_used", "")), context_text
                    )
            else:
                critique_payload = await self._fallback_html_critic(
                    str(state.get("html_used", "")), context_text
                )

            score_raw = critique_payload.get("score", 6)
            try:
                score = int(score_raw)
            except Exception:
                score = 6
            all_scores.append(score)

            issues = critique_payload.get("issues", [])
            suggestions = critique_payload.get("suggestions", [])
            if isinstance(issues, list):
                all_issues.extend([str(i) for i in issues if str(i).strip()])
            if isinstance(suggestions, list):
                all_suggestions.extend([str(s) for s in suggestions if str(s).strip()])

            await self.event_broker.publish(
                {
                    "type": "critic_artboard_scored",
                    "artboard_id": artboard_id,
                    "score": score,
                    "via": "gemini" if gemini_used else "groq_fallback",
                }
            )

        overall_score = min(all_scores) if all_scores else 6
        unique_issues = list(dict.fromkeys(all_issues))
        unique_suggestions = list(dict.fromkeys(all_suggestions))

        next_action = "design_review"
        if overall_score < target_score and current_round < max_rounds:
            next_action = "refine"

        critique = {
            "score": overall_score,
            "issues": unique_issues,
            "suggestions": unique_suggestions,
        }

        await self.event_broker.publish(
            {
                "type": "critic_completed",
                "score": overall_score,
                "issues": unique_issues,
                "next_action": next_action,
            }
        )

        return {
            **state,
            "critique": critique,
            "next_action": next_action,
            "round": current_round + 1,
            "stage": "critic_done",
        }

    @traceable
    async def refine_node(self, state: PipelineState) -> PipelineState:
        artboard_ids = state.get("artboard_ids", []) or []
        critique = state.get("critique", {}) or {}
        suggestions = critique.get("suggestions", []) if isinstance(critique, dict) else []
        feedback_instruction = str(state.get("feedback_instruction", "") or state.get("user_feedback", ""))

        website_content = state.get("website_content", {}) or {}
        palette = state.get("approved_palette", {}) or {}

        updated_html_list: list[str] = []
        updated_html_by_artboard = dict(state.get("last_html_by_artboard", {}) or {})
        previous_html_by_artboard = dict(state.get("last_html_by_artboard", {}) or {})
        if not previous_html_by_artboard:
            previous_html_by_artboard = self._build_artboard_html_index(
                [str(a) for a in artboard_ids],
                str(state.get("html_used", "") or ""),
            )

        tool_names = await self._get_paper_tool_names() if self.paper_client else set()
        has_screenshot_tool = "get_screenshot" in tool_names
        has_update_styles_tool = "update_styles" in tool_names

        for artboard_id in artboard_ids:
            fallback_html = str(
                previous_html_by_artboard.get(artboard_id)
                or state.get("current_html")
                or state.get("html_used")
                or ""
            )
            current_html = await self._get_artboard_html_for_refine(artboard_id, fallback_html)

            # Fast path: for simple style-only requests (e.g., button corner radius),
            # patch nodes directly via update_styles and skip full HTML regeneration.
            if has_update_styles_tool and self._is_button_radius_request(feedback_instruction):
                patched = await self._apply_targeted_style_patch(artboard_id, feedback_instruction)
                if patched:
                    print("[Pipeline][refine] style-only refine applied via update_styles")
                    updated_html_list.append(current_html)
                    updated_html_by_artboard[str(artboard_id)] = current_html
                    continue

            combined_suggestions = [str(s) for s in suggestions] if isinstance(suggestions, list) else []

            if self.paper_client and has_screenshot_tool and self.gemini_api_key:
                try:
                    screenshot_result = await self.paper_client.invoke_tool(
                        "get_screenshot", {"nodeId": artboard_id}
                    )
                    screenshot_b64 = self._extract_image_base64_payload(screenshot_result)
                    if screenshot_b64:
                        vision_hints = await self._call_gemini_flash_refine_hints(
                            screenshot_b64=screenshot_b64,
                            user_feedback=feedback_instruction,
                            critique_suggestions=combined_suggestions,
                            current_html=current_html,
                        )
                        if vision_hints:
                            combined_suggestions.extend(vision_hints)
                            print(f"[Pipeline][refine] Gemini Flash hints applied: {len(vision_hints)}")
                except Exception as exc:
                    print(f"[Pipeline][refine] Gemini Flash hinting skipped: {exc}")

            combined_suggestions = list(dict.fromkeys([s for s in combined_suggestions if str(s).strip()]))
            suggestions_text = "\n".join([f"- {s}" for s in combined_suggestions])

            refine_prompt = (
                f"Current HTML:\n{current_html}\n\n"
                f"Critique suggestions:\n{suggestions_text}\n\n"
                f"User feedback instruction:\n{feedback_instruction}\n\n"
                "You MUST apply user feedback exactly if provided.\n\n"
                f"Website content (must preserve factual content):\n{json.dumps(website_content)}\n"
                f"Approved palette (must remain consistent):\n{json.dumps(palette)}\n\n"
                'Return strict JSON only: {"summary": "...", "html": "..."}'
            )

            refined_html = ""
            try:
                response = await self.designer_llm.ainvoke(
                    [
                        SystemMessage(content=REFINE_SYSTEM_PROMPT),
                        HumanMessage(content=refine_prompt),
                    ]
                )
                payload = self._try_parse_json(getattr(response, "content", ""))
                if isinstance(payload, dict):
                    refined_html = str(payload.get("html", "")).strip()
            except Exception:
                refined_html = ""

            if not refined_html:
                refined_html = current_html or str(state.get("html_used", ""))

            if self._is_invalid_html_candidate(refined_html):
                print("[Refine Guard] Model returned non-HTML or tool-error text. Reusing previous HTML snapshot.")
                refined_html = current_html or str(state.get("html_used", ""))

            refined_html = self._enforce_full_bleed_html(refined_html)

            if self.paper_client:
                try:
                    await self._clear_artboard(artboard_id)
                except Exception as exc:
                    print(f"[Paper MCP] Could not clear artboard {artboard_id} before refine: {exc}")

                write_payload = {"html": refined_html, "targetNodeId": artboard_id, "mode": "insert-children"}
                print(
                    f"[Pipeline][refine] write_html requested: artboard={artboard_id}, round={state.get('round', 0)}, html_chars={len(refined_html)}"
                )
                
                write_result = await self.paper_client.invoke_tool(
                    "write_html",
                    write_payload,
                )
                created_count = len(self._extract_created_node_ids(write_result))
                self._log_paper_result(
                    "refine",
                    "write_html",
                    write_result,
                    detail=f"created_nodes={created_count}",
                )
                
                if write_result.get("isError"):
                    fallback_payload = {"html": refined_html, "targetNodeId": artboard_id, "mode": "replace"}
                    print("[Pipeline][refine] write_html fallback: trying mode=replace")
                    
                    fallback_result = await self.paper_client.invoke_tool(
                        "write_html",
                        fallback_payload,
                    )
                    fallback_count = len(self._extract_created_node_ids(fallback_result))
                    self._log_paper_result(
                        "refine",
                        "write_html(fallback)",
                        fallback_result,
                        detail=f"created_nodes={fallback_count}",
                    )

            updated_html_list.append(refined_html)
            updated_html_by_artboard[str(artboard_id)] = refined_html

        await self.event_broker.publish(
            {"type": "refine_completed", "round": state.get("round", 0)}
        )

        return {
            **state,
            "last_html_by_artboard": updated_html_by_artboard,
            "html_used": "\n\n<!-- refined-artboard -->\n\n".join(updated_html_list) if updated_html_list else state.get("html_used", ""),
            "current_html": updated_html_list[-1] if updated_html_list else state.get("current_html", ""),
            "stage": "critic",
        }

    @traceable
    async def design_review_node(self, state: PipelineState) -> PipelineState:
        critique = state.get("critique", {}) or {}
        score = int(critique.get("score", 0) or 0)

        friendly_message = await self._generate_friendly_message(
            f"The design is complete with a critic score of {score}/10. "
            "Tell the user it looks great and ask if they want any changes or if they are happy to export."
        )

        user_response = interrupt(
            {
                "type": "design_review",
                "artboard_ids": state.get("artboard_ids", []),
                "score": score,
                "message": friendly_message,
                "options": ["Looks great, export it!", "I want some changes"],
            }
        )

        response_text = ""
        if isinstance(user_response, dict):
            for key in ["response", "feedback", "choice", "message", "text"]:
                value = user_response.get(key)
                if isinstance(value, str) and value.strip():
                    response_text = value.strip()
                    break
        if not response_text:
            response_text = str(user_response).strip()

        classification_prompt = (
            "Classify the user's design review response into one action: done, refine, redesign.\n"
            "Rules:\n"
            "- done/export/looks good/perfect -> done\n"
            "- any change request -> refine\n"
            "- start over/different style -> redesign\n"
            'Return strict JSON only: {"action": "done|refine|redesign"}.\n\n'
            f"User response: {response_text}"
        )

        action = "refine"
        try:
            llm_response = await self.fast_llm.ainvoke(
                [
                    SystemMessage(content="You classify review intent."),
                    HumanMessage(content=classification_prompt),
                ]
            )
            payload = self._try_parse_json(getattr(llm_response, "content", ""))
            if isinstance(payload, dict):
                parsed_action = str(payload.get("action", "refine")).strip().lower()
                if parsed_action in {"done", "refine", "redesign"}:
                    action = parsed_action
        except Exception:
            lowered = response_text.lower()
            if any(token in lowered for token in ["export", "looks good", "perfect", "done"]):
                action = "done"
            elif any(token in lowered for token in ["start over", "different style", "redesign"]):
                action = "redesign"

        await self.event_broker.publish(
            {"type": "design_review_feedback", "feedback_action": action}
        )

        return {
            **state,
            "feedback_action": action,
            "user_feedback": response_text,
            "stage": "design_review",
        }

    @traceable
    async def feedback_node(self, state: PipelineState) -> PipelineState:
        user_feedback = str(state.get("user_feedback", "")).strip()
        extract_prompt = (
            "Extract a specific design instruction from user feedback. Return JSON:\n"
            "{\n"
            "  'scope': 'section|color|typography|layout|full',\n"
            "  'instruction': 'specific actionable change description',\n"
            "  'target_section': 'which section to change or null for all'\n"
            "}\n\n"
            f"User feedback: {user_feedback}"
        )

        scope = "layout"
        instruction = user_feedback or "Improve visual hierarchy and clarity."

        try:
            response = await self.fast_llm.ainvoke(
                [
                    SystemMessage(content="You extract actionable design instructions."),
                    HumanMessage(content=extract_prompt),
                ]
            )
            payload = self._try_parse_json(getattr(response, "content", ""))
            if isinstance(payload, dict):
                scope = str(payload.get("scope", scope)).strip().lower() or scope
                instruction = str(payload.get("instruction", instruction)).strip() or instruction
        except Exception:
            pass

        lowered = user_feedback.lower()
        redesign_requested = scope == "full" or any(
            token in lowered for token in ["start over", "redesign", "different style"]
        )

        feedback_action = "redesign" if redesign_requested else "refine"

        critique = state.get("critique", {}) if isinstance(state.get("critique", {}), dict) else {}
        suggestions = critique.get("suggestions", []) if isinstance(critique.get("suggestions", []), list) else []
        updated_critique = {
            **critique,
            "suggestions": list(dict.fromkeys([*suggestions, instruction])),
        }

        await self.event_broker.publish(
            {"type": "feedback_parsed", "scope": scope, "instruction": instruction}
        )

        updated_state: PipelineState = {
            **state,
            "feedback_action": feedback_action,
            "feedback_instruction": instruction,
            "critique": updated_critique,
            "stage": "feedback_parsed",
        }
        if feedback_action == "refine":
            updated_state["round"] = 0
        return updated_state

    @traceable
    async def export_node(self, state: PipelineState) -> PipelineState:
        artboard_ids = state.get("artboard_ids", []) or []
        jsx_parts: list[str] = []

        for artboard_id in artboard_ids:
            if not self.paper_client:
                continue
            try:
                get_jsx_payload = {"nodeId": artboard_id, "format": "inline-styles"}
                print(f"[Pipeline][export] get_jsx requested: artboard={artboard_id}")
                
                jsx_result = await self.paper_client.invoke_tool(
                    "get_jsx",
                    get_jsx_payload,
                )
                self._log_paper_result("export", "get_jsx", jsx_result)
                
                jsx_text = self._normalize_text_payload(self._extract_primary_text_payload(jsx_result))
                if jsx_text.strip():
                    print(f"[Pipeline][export] JSX extracted: artboard={artboard_id}, jsx_chars={len(jsx_text)}")
                    jsx_parts.append(f"// Artboard: {artboard_id}\n{jsx_text}")
            except Exception as e:
                print(f"\n❌ Error exporting JSX for {artboard_id}: {str(e)}\n")
                continue

        jsx_export = "\n\n".join(jsx_parts)

        website_content = state.get("website_content", {}) or {}
        critique = state.get("critique", {}) or {}
        plan = state.get("page_plan", {}) or {}
        plan_sections: list[str] = []
        if isinstance(plan, dict) and isinstance(plan.get("artboards"), list):
            for art in plan["artboards"]:
                if isinstance(art, dict) and isinstance(art.get("sections"), list):
                    plan_sections.extend([str(s) for s in art.get("sections", [])])

        summary_prompt = (
            "Summarize what was built in 2-3 sentences:\n"
            f"product={website_content.get('name', 'Unknown')}, "
            f"genre={state.get('genre', 'general')}, "
            f"rounds={state.get('round', 0)}, "
            f"final_score={critique.get('score', 0)}, "
            f"sections={list(dict.fromkeys(plan_sections))}"
        )

        design_summary = ""
        try:
            response = await self.fast_llm.ainvoke(
                [
                    SystemMessage(content="You write concise build summaries."),
                    HumanMessage(content=summary_prompt),
                ]
            )
            design_summary = str(getattr(response, "content", "")).strip()
        except Exception:
            design_summary = "Design session complete with exported artboards and structured content."

        # Log the final export summary
        print("\n" + "="*80)
        print("🎉 DESIGN SESSION COMPLETE - EXPORT SUMMARY")
        print("="*80)
        print("\n📄 Design Summary:")
        print("-"*80)
        print(design_summary)
        print("-"*80)
        
        jsx_chars = len(jsx_export) if isinstance(jsx_export, str) else 0
        print(f"\n📋 JSX Export ready: {'yes' if jsx_chars > 0 else 'no'} (chars={jsx_chars})")
        
        print("\n📊 Final Session Data:")
        print("-"*80)
        final_data = {
            "artboard_ids": artboard_ids,
            "final_score": critique.get("score", 0),
            "total_rounds": state.get("round", 0),
            "design_summary": design_summary,
        }
        print(json.dumps(final_data, indent=2))
        print("-"*80 + "\n")

        # Mark exported nodes as finalized in Paper when the capability exists.
        if self.paper_client and artboard_ids:
            tool_names = await self._get_paper_tool_names()
            if "finish_working_on_nodes" in tool_names:
                finish_payloads = [
                    {"nodeIds": artboard_ids},
                    {"ids": artboard_ids},
                ]
                for payload in finish_payloads:
                    try:
                        finish_result = await self.paper_client.invoke_tool(
                            "finish_working_on_nodes",
                            payload,
                        )
                        if not finish_result.get("isError"):
                            print("[Paper MCP] finish_working_on_nodes completed.")
                            break
                    except Exception as exc:
                        print(f"[Paper MCP] finish_working_on_nodes failed: {exc}")
                else:
                    print("[Paper MCP] finish_working_on_nodes unavailable for provided payload shapes.")

        await self.event_broker.publish(
            {
                "type": "session_complete",
                "jsx_export": jsx_export,
                "design_summary": design_summary,
                "artboard_ids": artboard_ids,
                "final_score": critique.get("score", 0),
            }
        )

        return {
            **state,
            "jsx_export": jsx_export,
            "design_summary": design_summary,
            "done": True,
            "stage": "complete",
        }

    @staticmethod
    def _route_after_critic(state: PipelineState) -> str:
        return str(state.get("next_action", "design_review"))

    @staticmethod
    def _route_after_design_review(state: PipelineState) -> str:
        action = str(state.get("feedback_action", "refine"))
        if action in {"done", "refine", "redesign"}:
            return action
        return "refine"

    @staticmethod
    def _route_after_feedback(state: PipelineState) -> str:
        action = str(state.get("feedback_action", "refine"))
        if action == "redesign":
            return "redesign"
        return "refine"

    # --- Graph Construction ---
    def _build_graph(self):
        """Build the full workflow with post-content design and review stages."""
        workflow = StateGraph(PipelineState)

        workflow.add_node("intake_node", self.intake_node)
        workflow.add_node("intake_confirm_node", self.intake_confirm_node)
        workflow.add_node("palette_node", self.palette_node)
        workflow.add_node("palette_confirm_node", self.palette_confirm_node)
        workflow.add_node("content_gather_node", self.content_gather_node)
        workflow.add_node("content_confirm_node", self.content_confirm_node)
        workflow.add_node("section_planner_node", self.section_planner_node)
        workflow.add_node("designer_node", self.designer_node)
        workflow.add_node("critic_node", self.critic_node)
        workflow.add_node("refine_node", self.refine_node)
        workflow.add_node("design_review_node", self.design_review_node)
        workflow.add_node("feedback_node", self.feedback_node)
        workflow.add_node("export_node", self.export_node)

        workflow.add_edge(START, "intake_node")
        workflow.add_edge("intake_node", "intake_confirm_node")
        workflow.add_edge("intake_confirm_node", "palette_node")
        workflow.add_edge("palette_node", "palette_confirm_node")
        workflow.add_edge("palette_confirm_node", "content_gather_node")
        workflow.add_edge("content_gather_node", "content_confirm_node")
        workflow.add_edge("content_confirm_node", "section_planner_node")
        workflow.add_edge("section_planner_node", "designer_node")
        workflow.add_edge("designer_node", "critic_node")
        workflow.add_edge("refine_node", "critic_node")
        workflow.add_edge("export_node", END)

        workflow.add_conditional_edges(
            "critic_node",
            self._route_after_critic,
            {
                "refine": "refine_node",
                "design_review": "design_review_node",
            },
        )

        workflow.add_conditional_edges(
            "design_review_node",
            self._route_after_design_review,
            {
                "done": "export_node",
                "refine": "feedback_node",
                "redesign": "section_planner_node",
            },
        )

        workflow.add_conditional_edges(
            "feedback_node",
            self._route_after_feedback,
            {
                "refine": "refine_node",
                "redesign": "section_planner_node",
            },
        )

        return workflow.compile(
            checkpointer=MemorySaver(),
            interrupt_before=[
                "intake_confirm_node",
                "palette_confirm_node",
                "content_confirm_node",
                "design_review_node",
            ],
        )

    # --- Execution Method ---
    async def run_discovery(self, session_id: str, brief: str):
        """Entry point to trigger the discovery pipeline up to content confirmation."""
        print(f"\n[System] Starting discovery for session {session_id[:8]}...")
        
        initial_state: PipelineState = {
            "session_id": session_id,
            "source": "text",
            "brief": brief,
            "genre": None,
            "emotional_direction": None,
            "intake_questions": [],
            "intake_answers": {},
            "palettes": [],
            "palette_artboard_id": None,
            "approved_palette": None,
            "content_questions": [],
            "content_answers": {},
            "website_content": {},
            "page_plan": {},
            "artboard_ids": [],
            "last_html_by_artboard": {},
            "node_ids": [],
            "html_used": "",
            "current_html": "",
            "round": 0,
            "max_rounds": 3,
            "target_score": 8,
            "critique": {},
            "next_action": "design_review",
            "user_feedback": "",
            "feedback_action": "refine",
            "jsx_export": "",
            "design_summary": "",
            "done": False,
            "stage": "start",
            "user_raw_answers": {},
        }
        
        return await self.graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": session_id}},
        )
    #utility Functions
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
    async def _get_child_node_ids(self, node_id: str) -> list[str]:
        if not self.paper_client:
            return []
        
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

    async def _clear_artboard(self, artboard_id: str) -> None:
        if not self.paper_client:
            return
        
        child_ids = await self._get_child_node_ids(artboard_id)
        if child_ids:
            await self.paper_client.invoke_tool("delete_nodes", {"nodeIds": child_ids})

    @staticmethod
    def _build_palette_html(
        brief: str,
        palettes: list[dict[str, Any]],
        website_content: dict[str, Any] | None = None,
    ) -> str:
        """Render palette exploration with real mini website mockups."""
    
        def hex_to_rgb(h: str) -> str:
            h = h.lstrip("#")
            try:
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return f"{r},{g},{b}"
            except (ValueError, IndexError):
                return "99,102,241"
    
        # Use real content if available, fall back gracefully
        content = website_content or {}
        product_name = (
            content.get("name")
            or content.get("product_name")
            or "Your Product"
        )
        tagline = (
            content.get("tagline")
            or "A quick preview of this palette in action."
        )
        features = content.get("features") or []
        feature_1 = features[0] if len(features) > 0 else "Core Feature"
        feature_2 = features[1] if len(features) > 1 else "Key Benefit"
    
        # Personality badge styles
        personality_styles = {
            "expected": ("Expected", "#3b82f6", "#0c1a2e"),
            "bold":     ("Bold",     "#f59e0b", "#1a1000"),
            "refined":  ("Refined",  "#a78bfa", "#12001a"),
        }
    
        cards = ""
        for i, palette in enumerate(palettes[:3]):
            name        = str(palette.get("name", f"Palette {i + 1}"))
            description = str(palette.get("description", ""))
            personality = str(palette.get("personality", "")).lower()
            typo_weight = str(palette.get("typography_weight", "balanced"))
            colors      = palette.get("colors") or {}
    
            # Read all tokens with fallbacks
            bg              = str(colors.get("bg",             "#0a0a0a"))
            bg_alt          = str(colors.get("bg_alt",         "#0f0f0f"))
            surface         = str(colors.get("surface",        "#1a1a1a"))
            accent          = str(colors.get("accent",         "#6366f1"))
            accent_muted    = str(colors.get("accent_muted",   f"rgba({hex_to_rgb(accent)},0.15)"))
            text_color      = str(colors.get("text",           "#ffffff"))
            text_muted      = str(colors.get("text_muted",     "#888888"))
            border          = str(colors.get("border",         "#2a2a2a"))
            gradient_start  = str(colors.get("gradient_start", bg))
            gradient_end    = str(colors.get("gradient_end",   bg_alt))
            button_text     = str(colors.get("button_text",    "#ffffff"))
            border_rgba     = f"rgba({hex_to_rgb(border)},0.5)"
    
            # Typography weight → font-weight values
            weight_map = {
                "heavy":    ("900", "700"),
                "balanced": ("800", "600"),
                "light":    ("700", "500"),
            }
            hero_weight, card_weight = weight_map.get(typo_weight, ("800", "600"))
    
            # Personality badge
            badge_label, badge_text, badge_bg = personality_styles.get(
                personality, ("", accent, bg)
            )
    
            cards += f"""
    <div style="border-radius:20px;background:{bg};border:1px solid {border_rgba};
                display:flex;flex-direction:column;overflow:hidden;
                box-shadow:0 16px 40px rgba(0,0,0,0.4);">
    
      <!-- Palette header -->
      <div style="padding:16px 20px;border-bottom:1px solid {border_rgba};
                  display:flex;align-items:center;justify-content:space-between;
                  background:{bg_alt};">
        <div style="font-size:15px;font-weight:800;color:{accent};">{name}</div>
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;
                      text-transform:uppercase;color:{badge_text};
                      background:{badge_bg};border:1px solid {border_rgba};
                      padding:2px 8px;border-radius:999px;">{badge_label}</div>
          <div style="font-size:10px;color:{text_muted};">{typo_weight}</div>
        </div>
      </div>
    
      <!-- Mini website preview -->
    
      <!-- Nav mockup -->
      <div style="padding:10px 20px;background:{bg};border-bottom:1px solid {border_rgba};
                  display:flex;justify-content:space-between;align-items:center;">
        <div style="font-size:12px;font-weight:800;color:{accent};">{product_name[:12]}</div>
        <div style="display:flex;gap:10px;">
          <div style="font-size:9px;color:{text_muted};">Features</div>
          <div style="font-size:9px;color:{text_muted};">Pricing</div>
          <div style="background:{accent};color:{button_text};font-size:9px;
                      font-weight:700;padding:3px 8px;border-radius:4px;">Sign Up</div>
        </div>
      </div>
    
      <!-- Hero mockup with real gradient -->
      <div style="padding:28px 20px;
                  background:linear-gradient(135deg,{gradient_start} 0%,{gradient_end} 100%);
                  display:flex;flex-direction:column;align-items:center;
                  text-align:center;gap:10px;">
        <div style="font-size:18px;font-weight:{hero_weight};color:{text_color};
                    line-height:1.1;max-width:240px;">{product_name}</div>
        <div style="font-size:10px;color:{text_muted};max-width:200px;
                    line-height:1.5;">{tagline[:80]}</div>
        <div style="display:flex;gap:8px;margin-top:4px;">
          <div style="background:{accent};color:{button_text};padding:6px 14px;
                      border-radius:6px;font-size:10px;font-weight:700;">Get Started</div>
          <div style="border:1px solid {border_rgba};color:{text_muted};padding:6px 14px;
                      border-radius:6px;font-size:10px;">Learn More</div>
        </div>
      </div>
    
      <!-- Feature cards mockup -->
      <div style="padding:16px 20px;background:{bg};
                  display:grid;grid-template-columns:1fr 1fr;gap:10px;">
        <div style="background:{surface};border:1px solid {border_rgba};
                    padding:12px;border-radius:10px;">
          <div style="width:20px;height:20px;background:{accent_muted};
                      border-radius:5px;margin-bottom:7px;
                      display:flex;align-items:center;justify-content:center;">
            <div style="width:8px;height:8px;background:{accent};border-radius:2px;"></div>
          </div>
          <div style="font-size:10px;font-weight:{card_weight};
                      color:{text_color};margin-bottom:4px;">{feature_1[:20]}</div>
          <div style="font-size:9px;color:{text_muted};line-height:1.4;">
            Core capability of {product_name[:10]}</div>
        </div>
        <div style="background:{surface};border:1px solid {border_rgba};
                    padding:12px;border-radius:10px;">
          <div style="width:20px;height:20px;background:{accent_muted};
                      border-radius:5px;margin-bottom:7px;
                      display:flex;align-items:center;justify-content:center;">
            <div style="width:8px;height:8px;background:{accent};border-radius:2px;"></div>
          </div>
          <div style="font-size:10px;font-weight:{card_weight};
                      color:{text_color};margin-bottom:4px;">{feature_2[:20]}</div>
          <div style="font-size:9px;color:{text_muted};line-height:1.4;">
            Built for your workflow</div>
        </div>
      </div>
    
      <!-- Palette description -->
      <div style="padding:12px 20px;background:{bg_alt};
                  border-top:1px solid {border_rgba};">
        <div style="font-size:10px;color:{text_muted};line-height:1.5;
                    font-style:italic;">{description}</div>
      </div>
    
      <!-- Color swatch row -->
      <div style="padding:10px 20px;display:flex;gap:6px;
                  background:{bg};border-top:1px solid {border_rgba};">
        {''.join([
            f'<div title="{tok}" style="width:18px;height:18px;border-radius:4px;'
            f'background:{col};border:1px solid {border_rgba};flex-shrink:0;"></div>'
            for tok, col in [
                ("bg", bg), ("surface", surface), ("accent", accent),
                ("text", text_color), ("muted", text_muted)
            ]
        ])}
      </div>
    </div>"""
    
        return f"""<div style="width:100%;min-height:100vh;display:flex;flex-direction:column;
                        padding:48px;background:#050505;
                        font-family:Inter,system-ui,sans-serif;box-sizing:border-box;">
      <div style="text-align:center;margin-bottom:40px;">
        <div style="font-size:11px;letter-spacing:0.2em;text-transform:uppercase;
                    color:#444;margin-bottom:10px;">Choose Your Direction</div>
        <div style="font-size:28px;font-weight:800;color:#ffffff;">
          3 Palette Directions for {product_name}</div>
        <div style="font-size:14px;color:#555;margin-top:8px;">
          Pick one — you can refine colors after</div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
                  gap:28px;max-width:1200px;margin:0 auto;width:100%;">
        {cards.strip()}
      </div>
    </div>""".strip()
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
    def _normalize_text_payload(text: str) -> str:
        value = text.strip()
        if not value:
            return ""
        parsed = VibeframeAgentPipeline._try_parse_json(value)
        if isinstance(parsed, str):
            return parsed.strip()
        return value

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
    async def check_paper_connection(paper_client: Optional[PaperMCPClient]) -> bool:
        """Check if Paper MCP client is connected and functional."""
        if not paper_client:
            return False
        
        try:
            await paper_client.initialize()
            await paper_client.list_tools()
            return True
        except Exception as e:
            print(f"  Paper connection error: {e}")
            return False


    
    
import asyncio
import uuid
from typing import Dict, Any

from app.config import settings

@traceable(name="vibeframe.discovery_cli", run_type="chain", tags=["vibeframe", "discovery", "cli"])
async def main():
    """Terminal CLI for Vibeframe Discovery Pipeline."""
    print("\n" + "="*60)
    print("🎨  VIBEFRAME v2 — Discovery Pipeline")
    print("="*60)
    
    # Check LangSmith tracing configuration
    print("\n[System] Checking telemetry & dependencies...")
    print("-" * 60)
    if settings.langsmith_tracing and settings.langsmith_api_key:
        print(f"✅ LangSmith Tracing: Enabled (project: {settings.langsmith_project})")
    else:
        print("⚠️  LangSmith Tracing: Disabled (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env)")

    if settings.mistral_api_key and ChatMistralAI is not None:
        print(f"✅ Mistral LLM: Ready (model: {settings.mistral_model})")
    elif settings.mistral_api_key and ChatMistralAI is None:
        print("⚠️  Mistral LLM: API key found, but langchain-mistralai is unavailable. Install requirements to enable Mistral.")
    else:
        print("⚠️  Mistral LLM: Disabled (set MISTRAL_API_KEY in .env to enable)")

    if settings.groq_api_key:
        print(f"✅ Groq LLM: Ready (fallback model: {settings.groq_model})")
    else:
        print("⚠️  Groq LLM: Disabled (set GROQ_API_KEY in .env to enable fallback)")
    
    # Initialize Paper client similar to backend/main.py startup flow.
    paper_client: Optional[PaperMCPClient] = PaperMCPClient()
    try:
        await paper_client.initialize()
        tools = await paper_client.list_tools()
        print(f"✅ Paper MCP: Connected ({len(tools)} tools available)")
    except Exception as exc:
        print(f"⚠️  Paper MCP: Not connected ({exc})")
        paper_client = None

    # Initialize pipeline
    event_broker = AgentEventBroker()
    pipeline = VibeframeAgentPipeline(
        groq_api_key=SecretStr(settings.groq_api_key),
        paper_client=paper_client,
        event_broker=event_broker
    )
    session_id = str(uuid.uuid4())
    session_tag = f"session:{session_id}"
    
    print(f"✅ Designer LLM: {pipeline.designer_backend_name}")
    print("✅ Event Broker: Ready")
    print(f"✅ Session ID: {session_id}")
    print("-" * 60)
    
    # Step 1: Get website brief
    print("\n[Step 1/5] Website Brief")
    print("-" * 60)
    user_brief = input("\n💭 Describe your website idea:\n> ").strip()
    if not user_brief:
        user_brief = "A modern SaaS platform for project management"
        print(f"(Using default: {user_brief})")
    
    print(f"\n[System] Processing brief: '{user_brief[:50]}...'")
    
    with ls.tracing_context(
        enabled=bool(settings.langsmith_tracing and settings.langsmith_api_key),
        project_name=settings.langsmith_project,
        tags=["vibeframe", "discovery", "cli", session_tag],
        metadata={"session_id": session_id, "workflow": "discovery_cli", "source": "terminal"},
    ):
        # Step 2: Run intake and ask adaptive questions one-by-one.
        print("\n[Step 2/5] Intake Analysis & Adaptive Questions")
        print("-" * 60)
        
        state = await pipeline.intake_node({
            "session_id": session_id,
            "brief": user_brief,
            "stage": "start"
        })
    
        print(f"✓ Genre detected: {state.get('genre', 'general')}")
        print(f"✓ Emotional direction: {state.get('emotional_direction', 'professional')}")
        
        print("\n[Adaptive Intake] I will ask one question at a time based on your previous answer.")
        intake_answers: dict[str, str] = {}
        intake_questions: list[str] = []
        for i in range(3):
            next_q = await pipeline.generate_adaptive_question(
                stage="intake",
                brief=user_brief,
                genre=state.get("genre", "general") or "general",
                emotional_direction=state.get("emotional_direction", "professional") or "professional",
                asked_questions=intake_questions,
                collected_answers=intake_answers,
                max_questions=3,
            )
            if bool(next_q.get("done")):
                break
            question = str(next_q.get("question", "")).strip()
            if not question:
                break
            key = str(next_q.get("key", f"intake_{i+1}")).strip() or f"intake_{i+1}"
            answer = input(f"\n[Q{i+1}] {question}\n> ").strip()
            intake_questions.append(question)
            intake_answers[key] = answer
        
        state["intake_answers"] = intake_answers
        state["stage"] = "palette"
        
        # Step 3: Generate palettes
        print("\n[Step 3/5] Palette Generation")
        print("-" * 60)
        
        palette_state = await pipeline.palette_node(state)
        palettes = palette_state.get("palettes", [])
        
        print(f"✓ Generated {len(palettes)} palettes:")
        for i, p in enumerate(palettes, 1):
            print(f"  {i}. {p.get('name', 'Unknown')} - {p.get('description', '')}")
        
        if palette_state.get("palette_artboard_id"):
            print(f"✓ Palette artboard created on Paper")
        else:
            print(f"⚠️  Palette artboard skipped (Paper client unavailable)")
        
        # Step 4: Palette confirmation (INTERRUPT)
        print("\n[INTERRUPT] Which palette do you prefer?")
        palette_names = [p.get("name", f"Palette {i+1}") for i, p in enumerate(palettes)]
        for i, name in enumerate(palette_names, 1):
            print(f"  {i}. {name}")
        
        while True:
            choice = input("\nEnter palette number (1-3):\n> ").strip()
            try:
                palette_idx = int(choice) - 1
                if 0 <= palette_idx < len(palettes):
                    break
                print("Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("Please enter a number.")
        
        palette_state["approved_palette"] = palettes[palette_idx]
        palette_state["stage"] = "content_gather"
        
        print(f"✓ Selected: {palettes[palette_idx].get('name')}")
        
        # Step 5: Content gathering (adaptive)
        print("\n[Step 4/5] Content Questions (Adaptive)")
        print("-" * 60)
        
        content_state = await pipeline.content_gather_node(palette_state)

        print("\n[Adaptive Content] I will ask one question at a time and adapt from your previous answer.")
        content_answers: dict[str, str] = {}
        content_questions: list[str] = []
        for i in range(4):
            next_q = await pipeline.generate_adaptive_question(
                stage="content",
                brief=user_brief,
                genre=content_state.get("genre", "general") or "general",
                emotional_direction=content_state.get("emotional_direction", "professional") or "professional",
                asked_questions=content_questions,
                collected_answers=content_answers,
                max_questions=4,
                prior_answers=intake_answers,
                prior_questions=intake_questions,
            )
            if bool(next_q.get("done")):
                break
            question = str(next_q.get("question", "")).strip()
            if not question:
                break
            key = str(next_q.get("key", f"content_{i+1}")).strip() or f"content_{i+1}"
            answer = input(f"\n[Q{i+1}] {question}\n> ").strip()
            content_questions.append(question)
            content_answers[key] = answer

        website_content = pipeline.build_website_content_from_answers(
            content_answers,
            intake_answers=intake_answers,
        )
        website_content = await pipeline.enrich_website_content(
            brief=user_brief,
            genre=content_state.get("genre", "general") or "general",
            emotional_direction=content_state.get("emotional_direction", "professional") or "professional",
            intake_answers=intake_answers,
            content_answers=content_answers,
            website_content=website_content,
        )
        content_completeness = pipeline.summarize_content_completeness(
            website_content,
            intake_answers=intake_answers,
            content_answers=content_answers,
        )
        await event_broker.publish(
            {
                "type": "content_collected",
                "product_name": website_content.get("name", "Your Product"),
            }
        )
        final_state = {
            **content_state,
            "content_questions": content_questions,
            "content_answers": content_answers,
            "website_content": website_content,
            "content_completeness": content_completeness,
            "page_plan": {},
            "artboard_ids": [],
            "last_html_by_artboard": {},
            "node_ids": [],
            "html_used": "",
            "current_html": "",
            "round": 0,
            "max_rounds": 3,
            "target_score": 8,
            "critique": {},
            "next_action": "design_review",
            "user_feedback": "",
            "feedback_action": "refine",
            "feedback_instruction": "",
            "jsx_export": "",
            "design_summary": "",
            "done": False,
            "stage": "section_planning",
        }
    
    # Step 7: Summary
    print("\n[Step 5/5] Discovery Complete ✓")
    print("=" * 60)
    
    print("\n📋 Discovery Summary:")
    print(f"  Genre: {final_state.get('genre', 'N/A')}")
    approved = final_state.get('approved_palette')
    palette_name = approved.get('name', 'N/A') if approved else 'N/A'
    print(f"  Palette: {palette_name}")
    
    website_content = final_state.get("website_content", {})
    print(f"  Product: {website_content.get('name', 'N/A')}")
    print(f"  Tagline: {website_content.get('tagline', 'N/A')}")
    print(f"  Audience: {website_content.get('audience', 'N/A')}")
    print(f"  Features Count: {len(website_content.get('features', [])) if isinstance(website_content.get('features', []), list) else 0}")

    completeness = final_state.get("content_completeness", {})
    if completeness:
        print("\n📊 Content Completeness:")
        print(f"  Sources: {json.dumps(completeness.get('field_sources', {}))}")
        print(f"  Missing: {completeness.get('missing_fields', [])}")
        print(f"  Has sample copy: {completeness.get('has_sample_copy', False)}")

    print(f"\n🎉 Fun fact: Your captured website content is {json.dumps(website_content, indent=2)}")
    
    with ls.tracing_context(
        enabled=bool(settings.langsmith_tracing and settings.langsmith_api_key),
        project_name=settings.langsmith_project,
        tags=["vibeframe", "design", "cli", session_tag],
        metadata={"session_id": session_id, "workflow": "design_loop", "source": "terminal"},
    ):
        print("\n[Step 6/8] Section Planning")
        print("-" * 60)
        final_state = await pipeline.section_planner_node(final_state)
        plan = final_state.get("page_plan", {}) or {}
        planned = plan.get("artboards", []) if isinstance(plan, dict) else []
        print(f"✓ Planned {len(planned) if isinstance(planned, list) else 0} artboard(s)")

        print("\n[Step 7/8] Design Generation (Paper)")
        print("-" * 60)
        final_state = await pipeline.designer_node(final_state)
        artboard_ids = final_state.get("artboard_ids", []) or []
        print(f"✓ Built {len(artboard_ids)} artboard(s) in Paper")

        print("\n[Step 8/8] Critique, Refine, and Export")
        print("-" * 60)
        while True:
            final_state = await pipeline.critic_node(final_state)
            critique = final_state.get("critique", {}) or {}
            score = int(critique.get("score", 0) or 0)
            print(f"✓ Critic score: {score}/10")

            next_action = str(final_state.get("next_action", "design_review"))
            if next_action == "refine":
                print("↻ Auto-refining based on critic suggestions...")
                final_state = await pipeline.refine_node(final_state)
                continue

            user_choice = input(
                "\nDesign review: type 'done' to export, 'refine' for edits, or 'redesign' to rebuild pages\n> "
            ).strip().lower()

            if user_choice in {"done", "export", "looks good", "yes"}:
                final_state["feedback_action"] = "done"
                break

            if user_choice in {"redesign", "start over"}:
                final_state["feedback_action"] = "redesign"
                final_state = await pipeline.section_planner_node(final_state)
                final_state = await pipeline.designer_node(final_state)
                continue

            feedback_text = input("What should change?\n> ").strip()
            final_state["user_feedback"] = feedback_text
            final_state = await pipeline.feedback_node(final_state)
            if str(final_state.get("feedback_action", "refine")) == "redesign":
                final_state = await pipeline.section_planner_node(final_state)
                final_state = await pipeline.designer_node(final_state)
            else:
                final_state = await pipeline.refine_node(final_state)

        final_state = await pipeline.export_node(final_state)

    print("\n✅ Full pipeline complete (Discovery → Design → Export)")
    print(f"✓ Artboards: {len(final_state.get('artboard_ids', []) or [])}")
    print(f"✓ Final score: {(final_state.get('critique', {}) or {}).get('score', 0)}")
    print(f"✓ HTML length: {len(str(final_state.get('html_used', '') or ''))}")
    print(f"✓ JSX length: {len(str(final_state.get('jsx_export', '') or ''))}")
    print("=" * 60 + "\n")
    
    # Ensure LangSmith traces are submitted before exiting (important for background tracing)
    if settings.langsmith_tracing and settings.langsmith_api_key:
        try:
            from langchain_core.tracers.langchain import wait_for_all_tracers
            print("[LangSmith] Waiting for traces to be submitted...")
            wait_for_all_tracers()
            print("[LangSmith] ✓ All traces submitted successfully")
        except Exception as e:
            print(f"[LangSmith] Warning: Could not wait for traces: {e}")
    
    return final_state

if __name__ == "__main__":
    asyncio.run(main())