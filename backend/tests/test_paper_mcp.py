import unittest
from unittest.mock import patch

from app.paper_mcp import PaperMCPClient


class DummyResponse:
    def __init__(self, *, headers: dict[str, str], body: str | None = None, json_data=None, status_code: int = 200) -> None:
        self.headers = headers
        self.text = body or ""
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self):
        if self._json_data is None:
            raise ValueError("not json")
        return self._json_data


class DummyAsyncClient:
    def __init__(self, response: DummyResponse) -> None:
        self.response = response
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, headers):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return self.response


class PaperMCPClientTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_event_stream_prefers_matching_request_id(self) -> None:
        response = DummyResponse(
            headers={"content-type": "text/event-stream"},
            body=(
                "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{\"message\":\"working\"}}\n\n"
                "data: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"tools\":[{\"name\":\"paper.create\"}]}}\n\n"
            ),
        )

        parsed = PaperMCPClient()._parse_response_payload(response, 7)

        self.assertEqual(parsed["id"], 7)
        self.assertEqual(parsed["result"]["tools"][0]["name"], "paper.create")

    async def test_list_tools_handles_event_stream_responses(self) -> None:
        response = DummyResponse(
            headers={"content-type": "text/event-stream"},
            body="data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[{\"name\":\"paper.create\"}]}}\n\n",
        )

        with patch("app.paper_mcp.httpx.AsyncClient", return_value=DummyAsyncClient(response)):
            client = PaperMCPClient()
            client._initialized = True

            tools = await client.list_tools()

        self.assertEqual(tools[0]["name"], "paper.create")

    def test_headers_include_protocol_version(self) -> None:
        headers = PaperMCPClient()._headers()

        self.assertEqual(headers["MCP-Protocol-Version"], "2025-06-18")
        self.assertEqual(headers["Accept"], "application/json, text/event-stream")


if __name__ == "__main__":
    unittest.main()