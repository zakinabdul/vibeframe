from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH, env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Vibeframe Backend"
    app_env: str = "development"
    app_port: int = 8000

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    mistral_api_key: str = Field(default="", alias="MISTRAL_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_critic_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_CRITIC_MODEL")
    gemini_api_base: str = Field(default="https://generativelanguage.googleapis.com/v1beta", alias="GEMINI_API_BASE")

    paper_mcp_url: str = Field(default="http://127.0.0.1:29979/mcp", alias="PAPER_MCP_URL")
    paper_mcp_timeout_seconds: float = Field(default=20.0, alias="PAPER_MCP_TIMEOUT_SECONDS")
    paper_desktop_path: str = Field(default="", alias="PAPER_DESKTOP_PATH")


settings = Settings()
