import asyncio
import json
import os
from typing import Any, Literal, TypedDict

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from pydantic import SecretStr

try:
    from langchain_mistralai import ChatMistralAI
except Exception:
    ChatMistralAI = None  # type: ignore

from app.config import settings
from app.paper_mcp import PaperMCPClient, PaperMCPProtocolError

# Experimental v2 pipeline lives in agentv2.py. We expose it here so callers can
# trial new behavior without replacing the stable endpoint pipeline in this file.
try:
    from agentv2 import AgentEventBroker as AgentEventBrokerV2
    from agentv2 import PipelineState as PipelineStateV2
    from agentv2 import VibeframeAgentPipeline as VibeframeAgentPipelineV2
except Exception:
    AgentEventBrokerV2 = None  # type: ignore
    PipelineStateV2 = None  # type: ignore
    VibeframeAgentPipelineV2 = None  # type: ignore


# Ensure LangSmith environment variables are visible before any model calls.
if settings.langsmith_tracing:
    os.environ["LANGSMITH_TRACING"] = "true"
if settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
if settings.langsmith_project:
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
if settings.langsmith_endpoint:
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint


DESIGNER_SYSTEM_PROMPT = """You are Vibeframe's Principal Design Engineer. Your goal is to output a "Best-in-Class" landing page that looks like a high-end Dribbble or Linear.app concept.

CORE DESIGN PHILOSOPHY:
- Modern UI isn't just about layout; it's about depth, whitespace, and subtle gradients.
- Avoid 1990s web defaults. Use "Soft UI" or "Glassmorphism" where appropriate.
- Every section must feel like a deliberate "scene."

VISUAL POLISH RULES (CRITICAL):
1. BACKGROUNDS: Never use flat #ccc or #f7f7f7. Use ultra-subtle mesh gradients or "Surface" colors (e.g., #0A0A0B for dark or #FAFAFB for light).
2. SHADOWS: Use multi-layered soft shadows: `box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06)`.
3. BUTTONS: Add a subtle 1px top-border (white with 0.1 opacity) to buttons to give them a "3D" premium feel.
4. BORDERS: Use "Subtle Borders." Instead of #ccc, use `rgba(255,255,255,0.08)` for dark mode or `rgba(0,0,0,0.05)` for light.
5. RADIUS: Cards must have a minimum of 24px border-radius.

STRUCTURAL REQUIREMENTS:
- Use `display: flex` and `display: grid` exclusively. 
- HERO: Must be cinematic. Use a "Glow" effect (a radial gradient div behind the text) to create depth.
- TYPOGRAPHY: Use a system font stack: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'.
- SPACING: Use the "8pt Grid System." Everything should be multiples of 8 (16, 24, 32, 64, 128).
- FULL BLEED: Root container must be edge-to-edge (`width:100%`, `margin:0`, no outer max-width wrapper).

TECHNICAL CONSTRAINTS:
1. One root <div> only.
2. 100% Inline styles. No <style> tags.
3. Use `min-height: 100vh` for sections to ensure they feel full and intentional.
4. Do not use an outer page-level max-width container.

OUTPUT CONTRACT:
- Return strict JSON: {"summary": "Brief design rationale", "html": "Full HTML string"}
- No markdown formatting.
- The UI must be BEAUTIFUL, modern, and high-contrast.
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
- Sections exist organically to fit the product's genre (e.g. Nav and Hero are present, but other sections vary cleanly)
- Visual hierarchy is strong and readable
- Color system follows spec
- Spacing is generous and uncluttered
- Buttons and cards feel intentionally styled
- Overall page looks premium and coherent

Be concise and actionable.
"""


INTAKE_SYSTEM_PROMPT = """You are Vibeframe's Intake Agent. Analyze the user's website design brief.

1. Detect the website genre from: saas, ecommerce, portfolio, agency, hospitality, healthcare, finance, education, gaming, general
2. Generate exactly 2 concise, specific clarifying questions tailored to this product

Return strict JSON only:
{"genre": "string", "questions": ["question1", "question2"]}

Question rules:
- Questions must be specific to this product, not generic
- Focus on: target audience & tone, key sections/pages, desired feeling or mood
- Do NOT ask about colors or palette — that is handled separately
- Max 20 words per question

No markdown fences, no extra keys.
"""


PALETTE_SYSTEM_PROMPT = """You are Vibeframe's Palette Agent. Create 3 distinct, tailored color palette directions for a website.

Return strict JSON only:
{
  "palettes": [
    {
      "name": "short memorable name (2-3 words)",
      "description": "one sentence personality description",
      "colors": {
        "bg": "#hex",
        "accent": "#hex",
        "surface": "#hex",
        "text": "#hex",
        "muted": "#hex",
        "border": "#hex"
      }
    }
  ]
}

Rules:
- Each palette must be distinct in mood and color family
- Colors must match the website genre and brand context
- bg should be dark unless genre strongly calls for light (healthcare, education, etc.)
- accent must be vibrant and highly readable on bg background
- surface is slightly lighter than bg (for cards/panels)
- text must have contrast ratio > 4.5 on bg
- Be creative and genre-appropriate, avoid generic choices

No markdown fences, no extra keys.
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
    artboard_ids: list[str]
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
    stage: Literal["intake", "palette_requested", "content_gathering", "building", "done"]
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


class VibeframeAgentPipeline:
    def __init__(self, paper_client: PaperMCPClient, event_broker: AgentEventBroker, groq_api_key: str) -> None:
        self.paper_client = paper_client
        self.event_broker = event_broker
        self.gemini_api_key = settings.gemini_api_key
        self.gemini_critic_model = settings.gemini_critic_model
        self.gemini_api_base = settings.gemini_api_base.rstrip("/")
        secret_key = SecretStr(groq_api_key)
        self.designer_backend_name = "groq"
        self.designer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.35, api_key=secret_key)

        mistral_key = (settings.mistral_api_key or "").strip()
        mistral_model = (settings.mistral_model or "mistral-large-latest").strip()
        if ChatMistralAI is not None and mistral_key:
            try:
                self.designer_llm = ChatMistralAI(
                    model_name=mistral_model,
                    temperature=0.35,
                    api_key=SecretStr(mistral_key),  # type: ignore[arg-type]
                )
                self.designer_backend_name = "mistral"
            except Exception:
                self.designer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.35, api_key=secret_key)
                self.designer_backend_name = "groq"
        print(f"[Vibeframe][agents.py] Designer backend active: {self.designer_backend_name}")

        self.critic_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=secret_key)
        self._conversation_sessions: dict[str, ConversationSession] = {}
        self._gemini_circuit_open_until: float = 0.0  # epoch timestamp; Gemini skipped until this time
        self.graph = self._build_graph()

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

    @traceable(name="vibeframe.generate", run_type="chain", tags=["vibeframe", "generate"])
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

        # Inject approved palette colors directly into the brief as a design system spec
        base_brief = session.get("last_user_brief", brief)
        approved_palette = session.get("approved_palette", {})
        if approved_palette and isinstance(approved_palette, dict):
            colors = approved_palette.get("colors", {})
            palette_name = approved_palette.get("name", "Custom Palette")
            palette_desc = approved_palette.get("description", "")
            palette_spec = (
                f"\n\nApproved Design System — {palette_name}"
                + (f" ({palette_desc})" if palette_desc else "")
                + ":\n"
                + f"- Background: {colors.get('bg', '#0a0a0a')}\n"
                + f"- Primary accent: {colors.get('accent', '#6366f1')}\n"
                + f"- Surface/card: {colors.get('surface', '#1a1a1a')}\n"
                + f"- Main text: {colors.get('text', '#ffffff')}\n"
                + f"- Muted text: {colors.get('muted', '#888888')}\n"
                + f"- Borders: {colors.get('border', '#2a2a2a')}"
            )
            base_brief = f"{base_brief}{palette_spec}"


        website_content = session.get("website_content")
        if isinstance(website_content, dict):
            base_brief += (
                f"\n\nWEBSITE CONTENT TO USE:\n"
                f"Product name: {website_content.get('name', 'N/A')}\n"
                f"Tagline: {website_content.get('tagline', 'N/A')}\n"
                f"Features/Services: {website_content.get('features', 'N/A')}\n"
                f"Target audience: {website_content.get('audience', 'N/A')}"
            )

        # Pass the existing design artboard ID so the designer reuses it
        existing_artboard_id = session.get("design_artboard_id")
        initial_state: PipelineState = {
            "status": "initialized",
            "source": source,
            "brief": base_brief,
            "artboard_id": existing_artboard_id or None,
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
            "create_new": not bool(existing_artboard_id),
        }
        await self.event_broker.publish({"type": "generate_started", "source": source})
        try:
            result = await self.graph.ainvoke(initial_state)
            # Persist the design artboard_id in the session for future reuse
            if result.get("artboard_id"):
                session["design_artboard_id"] = result["artboard_id"]
            response = {
                "artboard_id": result.get("artboard_id"),
                "palette_artboard_id": session.get("palette_artboard_id"),
                "node_ids": result.get("node_ids", []),
                "html_used": result.get("html_used", ""),
                "round": result.get("round", 0),
                "critique": result.get("critique", {}),
                "done": result.get("done", False),
                "assistant_message": await self._generate_friendly_message(
                    "You just finished building the user's landing page! Tell them to check it out on the canvas and let you know if they want any changes."
                ),
                "questions": [],
                "conversation_stage": "done",
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

    @traceable(name="vibeframe.refine", run_type="chain", tags=["vibeframe", "refine"])
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
            # Use LLM intake agent to detect genre and generate contextual questions
            await self.event_broker.publish({"type": "intake_started", "conversation_id": session_id})
            intake_data = await self._ask_intake_agent(normalized_brief)
            genre = str(intake_data.get("genre", "general"))
            questions = intake_data.get("questions") or [
                "What is the primary goal of your website?",
                "Who is the target audience for this product?",
            ]
            session["genre"] = genre

            # Generate 3 custom palettes based on brief + genre via LLM
            await self.event_broker.publish({"type": "palette_generation_started", "genre": genre})
            palettes = await self._generate_palettes_for_brief(normalized_brief, genre)
            session["palettes"] = palettes

            # Create palette artboard on Paper canvas with the dynamic palettes
            palette_artboard_id = session.get("palette_artboard_id")
            if not palette_artboard_id:
                palette_artboard_id = await self._create_palette_artboard(normalized_brief, palettes)
                session["palette_artboard_id"] = palette_artboard_id

            palette_names = [p.get("name", f"Palette {i + 1}") for i, p in enumerate(palettes)]
            palette_list_str = ", ".join(f"'{n}'" for n in palette_names)
            session["stage"] = "palette_requested"
            session["questions"] = list(questions)

            await self.event_broker.publish(
                {
                    "type": "palette_artboard_created",
                    "conversation_id": session_id,
                    "artboard_id": palette_artboard_id,
                    "genre": genre,
                    "palette_names": palette_names,
                }
            )
            await self.event_broker.publish(
                {
                    "type": "clarifying_questions_sent",
                    "conversation_id": session_id,
                    "questions": questions,
                }
            )

            prompt = (
                f"The user wants to build a {genre} website. "
                f"You just created 3 color palettes for them on the Paper canvas: {palette_list_str}. "
                "Tell them to go check it out and pick one they like. "
                "Be natural and friendly, vary your phrasing each time."
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
                "questions": list(questions),
                "conversation_stage": "palette_requested",
            }

        if stage == "palette_requested":
            palettes = session.get("palettes") or []
            approved_index = self._extract_approved_palette_index(normalized_brief, palettes)

            if approved_index is None:
                if self._is_palette_approval(normalized_brief):
                    # Approval intent detected but no specific palette identified — ask to clarify
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

                # No approval intent at all
                palette_names = [p.get("name", f"Palette {i + 1}") for i, p in enumerate(palettes)]
                first_name = palette_names[0] if palette_names else "the first one"
                await self.event_broker.publish(
                    {"type": "palette_approval_required", "conversation_id": session_id, "source": source}
                )
                prompt = (
                    f"The user needs to choose a palette to proceed. "
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

            # Specific palette identified — store it and proceed to build
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
                f"You just approved a palette. Now you need real content from the user "
                f"to build their website. "
                "Ask them 3-4 specific questions to gather their product name, features, tagline, and audience. "
                "Ask them in a friendly, conversational way — not as a list, but naturally."
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
                
                extract_prompt = f"""Extract the website content from the user's response:
{normalized_brief}

Return strict JSON only matching this schema:
{{
  "name": "project name or null",
  "tagline": "tagline or null",
  "features": "features or null",
  "audience": "target audience or null"
}}"""
                extract_res = await self.designer_llm.ainvoke([HumanMessage(content=extract_prompt)])
                try:
                    parsed_content = self._try_parse_json(getattr(extract_res, "content", ""))
                    if isinstance(parsed_content, dict):
                        session["website_content"] = parsed_content
                except Exception:
                    pass
                
                session["stage"] = "building"
                # Emit an event to visibly acknowledge we've collected the content
                await self.event_broker.publish(
                    {
                        "type": "content_gathered",
                        "conversation_id": session_id,
                    }
                )
        return None

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
            if not name: continue
            if name in lowered: return i
            words = [w for w in name.split() if len(w) > 3]
            if words and words[0] in lowered: return i
        return None

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

    async def _create_palette_artboard(self, brief: str, palettes: list[dict[str, Any]]) -> str:
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

    @staticmethod
    def _build_palette_html(brief: str, palettes: list[dict[str, Any]]) -> str:
        """Render dynamic palette exploration HTML with mini website mockups."""
        def hex_to_rgb(h: str) -> str:
            h = h.lstrip("#")
            try:
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return f"{r},{g},{b}"
            except (ValueError, IndexError):
                return "99,102,241"

        product_name = "Your Project"
        words = [w for w in brief.split() if w.isalnum() or w.endswith(('.','!','?'))]
        if words:
            product_name = " ".join(words[:4]).title().replace('.', '').replace(',', '')

        cards = ""
        for i, palette in enumerate(palettes[:3]):
            name = str(palette.get("name", f"Palette {i + 1}"))
            colors = palette.get("colors") or {}
            accent = str(colors.get("accent", "#6366f1"))
            bg_card = str(colors.get("bg", "#0a0a0a"))
            surface = str(colors.get("surface", "#1a1a1a"))
            text_color = str(colors.get("text", "#ffffff"))
            muted = str(colors.get("muted", "#888888"))
            border = str(colors.get("border", "#2a2a2a"))
            border_rgba = f"rgba({hex_to_rgb(border)},0.35)"

            cards += f"""
    <div style="border-radius:16px;padding:24px;background:{bg_card};border:1px solid {border_rgba};display:flex;flex-direction:column;gap:16px;color:{text_color};overflow:hidden;box-shadow:0 12px 24px rgba(0,0,0,0.2);">
      <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid {border_rgba};padding-bottom:12px;">
        <div style="font-size:16px;font-weight:800;color:{accent};">{name}</div>
        <div style="font-size:12px;color:{muted};display:flex;gap:12px;"><span>Features</span><span>About</span></div>
      </div>
      <div style="text-align:center;padding:24px 0;display:flex;flex-direction:column;align-items:center;gap:12px;">
        <div style="font-size:24px;font-weight:900;line-height:1.1;">{product_name}</div>
        <div style="font-size:12px;color:{muted};max-width:200px;">A quick preview of how this palette feels on a real web layout.</div>
        <div style="background:{accent};color:{bg_card};padding:8px 16px;border-radius:6px;font-size:12px;font-weight:700;margin-top:8px;">Get Started</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;">
        <div style="background:{surface};border:1px solid {border_rgba};padding:14px;border-radius:8px;">
          <div style="width:24px;height:24px;background:{accent};border-radius:6px;margin-bottom:8px;"></div>
          <div style="font-size:12px;font-weight:700;margin-bottom:4px;">Feature One</div>
          <div style="font-size:10px;color:{muted};line-height:1.4;">Clean and simple card styling.</div>
        </div>
        <div style="background:{surface};border:1px solid {border_rgba};padding:14px;border-radius:8px;">
          <div style="width:24px;height:24px;border:1px solid {accent};border-radius:6px;margin-bottom:8px;"></div>
          <div style="font-size:12px;font-weight:700;margin-bottom:4px;">Feature Two</div>
          <div style="font-size:10px;color:{muted};line-height:1.4;">With proper contrast ratios.</div>
        </div>
      </div>
    </div>"""

        return f"""<div style="width:100%;min-height:100vh;display:flex;flex-direction:column;gap:32px;padding:48px;background:#050505;color:#ffffff;font-family:Inter,sans-serif;">
  <div style="display:flex;flex-direction:column;gap:8px;text-align:center;margin-bottom:16px;">
    <div style="font-size:12px;letter-spacing:0.15em;text-transform:uppercase;color:#888;">Design System</div>
    <div style="font-size:32px;font-weight:800;">Select Your Palette</div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:32px;flex:1;max-width:1200px;margin:0 auto;">{cards.strip()}
  </div>
</div>""".strip()

    async def _ask_intake_agent(self, brief: str) -> dict[str, Any]:
        """Use LLM to detect website genre and generate contextual clarifying questions."""
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
        """Use LLM to generate 3 custom color palettes tailored to the brief and genre."""
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

    async def _load_palette_context(self, palette_artboard_id: str | None) -> str:
        if not palette_artboard_id:
            return ""
        try:
            return await self._get_document_html(node_id=palette_artboard_id)
        except Exception:
            return ""

    @staticmethod
    def _summarize_palette_context(_palette_context: str) -> str:
        return "Palette confirmed and applied to design system."

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


    async def _parse_section_intent(self, brief: str) -> dict[str, Any]:
        prompt = """Analyze the brief to see if the user explicitly wants multiple separate sections or pages as different artboards.
Examples:
- "three section website" -> {"multi_artboard": true, "artboard_names": ["Hero", "Features", "Contact"]}
- "landing page" -> {"multi_artboard": false, "artboard_names": []}
- "home about services contact pages" -> {"multi_artboard": true, "artboard_names": ["Home", "About", "Services", "Contact"]}

Brief: """ + brief + """
Return strict JSON only."""
        try:
            res = await self.designer_llm.ainvoke([HumanMessage(content=prompt)])
            parsed = self._try_parse_json(getattr(res, "content", ""))
            if isinstance(parsed, dict) and "multi_artboard" in parsed:
                return parsed
        except Exception:
            pass
        return {"multi_artboard": False, "artboard_names": []}

    async def _designer_node(self, state: PipelineState) -> PipelineState:
        await self.event_broker.publish({"type": "designer_started", "round": state.get("round", 0)})
        brief = state.get("brief", "")
        current_html = state.get("current_html", "")

        existing_artboard_id = state.get("artboard_id")
        create_new = state.get("create_new", True)

        design = await self._ask_designer(brief=brief, current_html=current_html, critique={})
        
        if create_new or not existing_artboard_id:
            create_payload = {
                "name": "Vibeframe App Canvas",
                "styles": {
                    "width": "1440px",
                    "height": "auto",
                    "minHeight": "900px",
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
        else:
            artboard_id = existing_artboard_id
            await self._clear_artboard(artboard_id)
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
            "artboard_ids": [artboard_id],
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
            import time as _time
            _now = _time.monotonic()
            if _now < self._gemini_circuit_open_until:
                # Circuit open — Gemini is rate-limited; skip to Groq immediately
                remaining = int(self._gemini_circuit_open_until - _now)
                await self.event_broker.publish(
                    {
                        "type": "critic_gemini_skipped",
                        "reason": "circuit_open",
                        "retry_in_seconds": remaining,
                    }
                )
            else:
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
                    # Trip the circuit breaker on 429; all other errors also fall back
                    if "429" in str(exc):
                        self._gemini_circuit_open_until = _time.monotonic() + 60.0
                    await self.event_broker.publish(
                        {
                            "type": "critic_provider_fallback",
                            "from": "gemini",
                            "to": "groq",
                            "message": str(exc),
                            "circuit_tripped": "429" in str(exc),
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
        await self._clear_artboard(artboard_id)

        # Ensure the artboard retains its canvas properties just in case
        await self.paper_client.invoke_tool(
            "update_styles",
            {
                "nodeIds": [artboard_id],
                "styles": {
                    "width": "1440px",
                    "height": "auto",
                    "minHeight": "900px",
                    "backgroundColor": "#0a0a0a",
                }
            }
        )

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

    async def _clear_artboard(self, artboard_id: str) -> None:
        child_ids = await self._get_child_node_ids(artboard_id)
        if child_ids:
            await self.paper_client.invoke_tool("delete_nodes", {"nodeIds": child_ids})

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
