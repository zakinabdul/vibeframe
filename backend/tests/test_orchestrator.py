import unittest

from app.orchestrator import DesignOrchestrator
from app.schemas import PaperAction


class StubDesigner:
    def __init__(self, actions: list[PaperAction]) -> None:
        self.actions = actions

    def plan_actions(self, prompt: str, tools):
        return "stub", self.actions


class StubPaperClient:
    def __init__(self) -> None:
        self.invocations = []

    async def list_tools(self):
        return [
            {"name": "create_artboard"},
            {"name": "write_html"},
            {"name": "set_text_content"},
            {"name": "update_styles"},
        ]

    async def invoke_tool(self, tool_name, arguments):
        self.invocations.append((tool_name, arguments))
        if tool_name == "create_artboard":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": '{"id":"2-0","name":"Generated Artboard"}',
                    }
                ]
            }
        return {"content": [{"type": "text", "text": "ok"}]}


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolves_last_artboard_placeholder(self):
        actions = [
            PaperAction(tool="create_artboard", arguments={"width": 1200, "height": 800}),
            PaperAction(
                tool="write_html",
                arguments={
                    "targetNodeId": "$LAST_ARTBOARD_ID",
                    "mode": "insert-children",
                    "html": '<div style="display:flex;width:100%;height:100%">Hello</div>',
                },
            ),
        ]
        paper = StubPaperClient()
        orchestrator = DesignOrchestrator(paper_client=paper, designer=StubDesigner(actions))

        response = await orchestrator.run("test")

        self.assertEqual(response.actions_executed, 2)
        self.assertEqual(paper.invocations[1][1]["targetNodeId"], "2-0")

    def test_normalize_set_text_content_legacy_shape(self):
        normalized = DesignOrchestrator._normalize_legacy_arguments(
            "set_text_content",
            {"node_id": 7, "text": "Welcome"},
        )

        self.assertIn("updates", normalized)
        self.assertEqual(normalized["updates"][0]["nodeId"], "7")
        self.assertEqual(normalized["updates"][0]["textContent"], "Welcome")

    def test_normalize_create_artboard_legacy_shape(self):
        normalized = DesignOrchestrator._normalize_legacy_arguments(
            "create_artboard",
            {"width": 1440, "height": 900, "background": "#000"},
        )

        self.assertEqual(normalized["name"], "Generated Artboard")
        self.assertEqual(normalized["styles"]["width"], "1440px")
        self.assertEqual(normalized["styles"]["height"], "900px")
        self.assertEqual(normalized["styles"]["backgroundColor"], "#000")


if __name__ == "__main__":
    unittest.main()