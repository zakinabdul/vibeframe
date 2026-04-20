from typing import Any, Literal

from pydantic import BaseModel, Field


class DesignRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)


class PaperAction(BaseModel):
    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class DesignResponse(BaseModel):
    summary: str
    actions_planned: int
    actions_executed: int
    tool_trace: list[dict[str, Any]]


class GenerateRequest(BaseModel):
    brief: str = Field(min_length=3, max_length=5000)
    source: Literal["text", "voice"] = "text"
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)


class Critique(BaseModel):
    score: int = 0
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    artboard_id: str | None = None
    palette_artboard_id: str | None = None
    node_ids: list[str] = Field(default_factory=list)
    html_used: str = ""
    round: int = 0
    critique: Critique = Field(default_factory=Critique)
    done: bool = False
    assistant_message: str = ""
    questions: list[str] = Field(default_factory=list)
    conversation_stage: str = "building"


class RefineRequest(BaseModel):
    artboard_id: str = Field(min_length=1)
    instruction: str = Field(min_length=3, max_length=5000)


class RefineResponse(BaseModel):
    artboard_id: str
    node_ids: list[str] = Field(default_factory=list)
    html_used: str
    mode_used: str | None = None
    tool_result: dict[str, Any] = Field(default_factory=dict)


class CanvasArtboard(BaseModel):
    id: str
    name: str


class CanvasCurrentResponse(BaseModel):
    source: str
    document_html: str
    artboards: list[CanvasArtboard] = Field(default_factory=list)
    raw: dict[str, Any] | None = None


class PaperOpenResponse(BaseModel):
    opened: bool = False
    message: str = ""


class ResetSessionRequest(BaseModel):
    conversation_id: str | None = None


class TranscribeResponse(BaseModel):
    text: str = ""
    provider: str = "groq"
    model: str = ""
