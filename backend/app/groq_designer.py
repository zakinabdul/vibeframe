import json
from typing import Any

from groq import Groq

from app.config import settings
from app.schemas import PaperAction

DESIGNER_SYSTEM_PROMPT = """You are Vibeframe Designer. Convert user UI intent into concrete Paper MCP tool calls.
Return strict JSON only with this schema:
{
  \"summary\": string,
  \"actions\": [
    { \"tool\": string, \"arguments\": object }
  ]
}
Rules:
- Output valid JSON only, no markdown.
- Prefer a small number of high-impact actions.
- Use the provided tool input schemas exactly. Never invent argument keys.
- Never guess node IDs. If you need the ID created by a prior create_artboard call, use the literal placeholder "$LAST_ARTBOARD_ID".
- Prefer create_artboard + write_html for fresh screen generation, because write_html can create a complete structure in one insertion.
- CRITICAL: for newly created UI content, include full inline styles directly in write_html HTML. Do not rely on update_styles for child elements.
- CRITICAL: update_styles and set_text_content only accept exact node IDs returned by previous tool results.
- CRITICAL: never use CSS selectors or pseudo paths as node IDs (examples of forbidden values: "$LAST_ARTBOARD_ID > h1", ".cta", "#title").
- For set_text_content and update_styles, always use the required updates[] schema.
- Never include unknown top-level keys.
"""


class GroqDesigner:
    def __init__(self) -> None:
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def plan_actions(self, prompt: str, tools: list[dict[str, Any]]) -> tuple[str, list[PaperAction]]:
        tool_names = [str(tool.get("name", "")) for tool in tools if tool.get("name")]
        tools_hint = ", ".join(tool_names) if tool_names else "No tools discovered"
        schema_hint = json.dumps(
            [
                {
                    "name": tool.get("name"),
                    "inputSchema": tool.get("inputSchema", {}),
                }
                for tool in tools
                if tool.get("name")
            ],
            ensure_ascii=True,
        )
        user_prompt = (
            f"User brief: {prompt}\n"
            f"Available Paper tools: {tools_hint}\n"
            f"Tool input schemas: {schema_hint}\n"
            "Generate actions that only use available tools and strictly follow each input schema."
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=1400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": DESIGNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = completion.choices[0].message.content or "{}"
        parsed: dict[str, Any] = json.loads(raw)
        summary = str(parsed.get("summary", "Generated a UI plan."))
        actions_raw = parsed.get("actions", [])

        actions: list[PaperAction] = []
        if isinstance(actions_raw, list):
            for item in actions_raw:
                if isinstance(item, dict):
                    try:
                        actions.append(PaperAction(**item))
                    except Exception:
                        continue

        return summary, actions
