from typing import Any
import json

from app.groq_designer import GroqDesigner
from app.paper_mcp import PaperMCPClient
from app.schemas import DesignResponse, PaperAction


class DesignOrchestrator:
    def __init__(self, paper_client: PaperMCPClient, designer: GroqDesigner) -> None:
        self.paper_client = paper_client
        self.designer = designer

    async def run(self, prompt: str) -> DesignResponse:
        tools = await self.paper_client.list_tools()
        allowed_tools = {str(tool.get("name", "")) for tool in tools if tool.get("name")}
        summary, planned_actions = self.designer.plan_actions(prompt=prompt, tools=tools)

        validated_actions = self._filter_actions(planned_actions, allowed_tools)
        tool_trace: list[dict[str, Any]] = []
        successful_executions = 0
        runtime_context: dict[str, str] = {}

        for action in validated_actions:
            normalized_arguments = self._normalize_legacy_arguments(action.tool, action.arguments)
            resolved_arguments = self._resolve_placeholders(normalized_arguments, runtime_context)

            if not self._arguments_usable_for_tool(action.tool, resolved_arguments):
                tool_trace.append(
                    {
                        "tool": action.tool,
                        "arguments": resolved_arguments,
                        "result": {
                            "isError": True,
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Skipped action because it used selector-like node IDs. Paper MCP requires exact node IDs returned from previous tool calls.",
                                }
                            ],
                        },
                    }
                )
                continue

            result = await self.paper_client.invoke_tool(action.tool, resolved_arguments)
            if not self._is_tool_error(result):
                successful_executions += 1

            if action.tool == "create_artboard":
                created_id = self._extract_created_node_id(result)
                if created_id:
                    runtime_context["LAST_ARTBOARD_ID"] = created_id

            tool_trace.append(
                {
                    "tool": action.tool,
                    "arguments": resolved_arguments,
                    "result": result,
                }
            )

        return DesignResponse(
            summary=summary,
            actions_planned=len(planned_actions),
            actions_executed=successful_executions,
            tool_trace=tool_trace,
        )

    @staticmethod
    def _filter_actions(actions: list[PaperAction], allowed_tools: set[str]) -> list[PaperAction]:
        valid: list[PaperAction] = []
        for action in actions:
            if action.tool in allowed_tools:
                valid.append(action)
        return valid

    @staticmethod
    def _normalize_legacy_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(arguments)

        if tool_name == "create_artboard" and "styles" not in normalized:
            width = normalized.pop("width", 1440)
            height = normalized.pop("height", 900)
            background = normalized.pop("background", None)
            styles: dict[str, Any] = {
                "width": f"{int(width)}px" if isinstance(width, (int, float)) else str(width),
                "height": f"{int(height)}px" if isinstance(height, (int, float)) else str(height),
            }
            if background and background != "none":
                styles["backgroundColor"] = background
            normalized["name"] = str(normalized.get("name") or "Generated Artboard")
            normalized["styles"] = styles
            normalized.pop("x", None)
            normalized.pop("y", None)

        if tool_name == "set_text_content" and "updates" not in normalized:
            node_id = normalized.pop("nodeId", normalized.pop("node_id", None))
            text_value = normalized.pop("textContent", normalized.pop("text", ""))
            if node_id is not None:
                normalized = {
                    "updates": [
                        {
                            "nodeId": str(node_id),
                            "textContent": str(text_value),
                        }
                    ]
                }

        if tool_name == "update_styles" and "updates" not in normalized:
            node_id = normalized.pop("nodeId", normalized.pop("node_id", None))
            style_map = normalized.pop("styles", {})
            legacy_style_fields = {
                "border_radius": "borderRadius",
                "background": "backgroundColor",
                "font_size": "fontSize",
                "align": "textAlign",
            }
            for key, mapped in legacy_style_fields.items():
                if key in normalized:
                    style_map[mapped] = normalized.pop(key)

            if node_id is not None:
                normalized = {
                    "updates": [
                        {
                            "nodeIds": [str(node_id)],
                            "styles": style_map,
                        }
                    ]
                }

        return normalized

    @staticmethod
    def _resolve_placeholders(value: Any, runtime_context: dict[str, str]) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            return runtime_context.get(value[1:], value)
        if isinstance(value, list):
            return [DesignOrchestrator._resolve_placeholders(item, runtime_context) for item in value]
        if isinstance(value, dict):
            return {k: DesignOrchestrator._resolve_placeholders(v, runtime_context) for k, v in value.items()}
        return value

    @staticmethod
    def _extract_created_node_id(result: dict[str, Any]) -> str | None:
        content = result.get("content")
        if not isinstance(content, list):
            return None

        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            node_id = parsed.get("id") if isinstance(parsed, dict) else None
            if isinstance(node_id, str) and node_id:
                return node_id
        return None

    @staticmethod
    def _is_tool_error(result: dict[str, Any]) -> bool:
        is_error = result.get("isError")
        return bool(is_error)

    @staticmethod
    def _arguments_usable_for_tool(tool_name: str, arguments: dict[str, Any]) -> bool:
        if tool_name not in {"set_text_content", "update_styles"}:
            return True

        updates = arguments.get("updates")
        if not isinstance(updates, list):
            return True

        for update in updates:
            if not isinstance(update, dict):
                continue

            node_ids: list[str] = []
            maybe_node_id = update.get("nodeId")
            if isinstance(maybe_node_id, str):
                node_ids.append(maybe_node_id)

            maybe_node_ids = update.get("nodeIds")
            if isinstance(maybe_node_ids, list):
                node_ids.extend([node for node in maybe_node_ids if isinstance(node, str)])

            for node_id in node_ids:
                if DesignOrchestrator._looks_like_selector(node_id):
                    return False

        return True

    @staticmethod
    def _looks_like_selector(value: str) -> bool:
        selector_tokens = [">", "#", ".", "[", "]", "(", ")"]
        return any(token in value for token in selector_tokens)
