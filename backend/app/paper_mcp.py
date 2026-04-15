from typing import Any
import json

import httpx

from app.config import settings


MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_ACCEPT_HEADER = "application/json, text/event-stream"


class PaperMCPError(Exception):
    pass


class PaperMCPConnectionError(PaperMCPError):
    pass


class PaperMCPProtocolError(PaperMCPError):
    pass


class PaperMCPClient:
    def __init__(self) -> None:
        self.url = settings.paper_mcp_url
        self.timeout = settings.paper_mcp_timeout_seconds
        self._session_id: str | None = None
        self._initialized = False
        self._request_id = 1

    def _headers(self, *, include_session: bool = True) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": MCP_ACCEPT_HEADER,
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        if include_session and self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    @staticmethod
    def _content_type(response: httpx.Response) -> str:
        return response.headers.get("content-type", "").split(";", 1)[0].strip().lower()

    @staticmethod
    def _parse_event_stream(content: str, request_id: int) -> dict[str, Any]:
        events: list[str] = []
        data_lines: list[str] = []

        def flush_event() -> None:
            if data_lines:
                events.append("\n".join(data_lines))
                data_lines.clear()

        for raw_line in content.splitlines():
            line = raw_line.rstrip("\r")
            if not line:
                flush_event()
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

        flush_event()

        last_parsed_message: dict[str, Any] | None = None
        for event in events:
            try:
                message = json.loads(event)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict):
                last_parsed_message = message
                if message.get("id") == request_id:
                    return message

        if last_parsed_message is not None:
            return last_parsed_message

        raise PaperMCPProtocolError("Paper MCP SSE response did not contain valid JSON-RPC data.")

    def _parse_response_payload(self, response: httpx.Response, request_id: int) -> dict[str, Any]:
        content_type = self._content_type(response)
        if content_type == "text/event-stream":
            return self._parse_event_stream(response.text, request_id)

        try:
            data = response.json()
        except ValueError as exc:
            body_preview = response.text[:200].replace("\n", "\\n")
            raise PaperMCPProtocolError(
                f"Paper MCP response was not valid JSON (content-type={content_type or 'unknown'}, body={body_preview!r})."
            ) from exc

        if not isinstance(data, dict):
            raise PaperMCPProtocolError("Paper MCP response payload is malformed.")

        return data

    async def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._initialized and method != "initialize":
            await self.initialize()

        request_id = self._request_id
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        include_session = method != "initialize"
        allow_stale_session_retry = include_session and bool(self._session_id)

        for attempt in range(2 if allow_stale_session_retry else 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.url, json=payload, headers=self._headers(include_session=include_session))
                    session_id = response.headers.get("mcp-session-id")
                    if session_id:
                        self._session_id = session_id
                    response.raise_for_status()
                    data = self._parse_response_payload(response, request_id)
                break
            except httpx.ConnectError as exc:
                raise PaperMCPConnectionError(
                    f"Cannot connect to Paper MCP at {self.url}. Ensure Paper Desktop MCP server is running."
                ) from exc
            except httpx.TimeoutException as exc:
                raise PaperMCPConnectionError(
                    f"Timed out talking to Paper MCP at {self.url}."
                ) from exc
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                body_preview = exc.response.text[:200].replace("\n", "\\n")
                is_stale_session = status_code in {400, 401, 403, 404, 409, 410}
                if allow_stale_session_retry and attempt == 0 and is_stale_session:
                    self._session_id = None
                    self._initialized = False
                    await self.initialize()
                    continue
                raise PaperMCPProtocolError(
                    f"Paper MCP returned HTTP {status_code}. body={body_preview!r}. "
                    "Check that Paper Desktop is running, a document is open, and PAPER_MCP_URL is correct."
                ) from exc

        if "error" in data:
            raise PaperMCPProtocolError(f"Paper MCP error: {data['error']}")

        result = data.get("result", {})
        if not isinstance(result, dict):
            raise PaperMCPProtocolError("Paper MCP result payload is malformed.")
        return result

    async def initialize(self) -> dict[str, Any]:
        result = await self._call(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "vibeframe",
                    "version": "0.1.0",
                },
            },
        )
        self._initialized = True
        await self._notify("notifications/initialized")
        return result

    async def _notify(self, method: str) -> None:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                await client.post(self.url, json=payload, headers=self._headers())
        except Exception:
            # Notification failures should not block tool execution attempts.
            return

    async def is_alive(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.post(
                    self.url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 0,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": MCP_PROTOCOL_VERSION,
                            "capabilities": {},
                            "clientInfo": {
                                "name": "vibeframe-ping",
                                "version": "0.0.1",
                            },
                        },
                    },
                    headers=self._headers(include_session=False),
                )
                return response.status_code in {200, 202}
        except Exception:
            return False

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._call("tools/list")
        tools = result.get("tools", [])
        return tools if isinstance(tools, list) else []

    async def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._call(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )
