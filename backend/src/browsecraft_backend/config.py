from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    browser_use_api_key: str | None = None
    github_token: str | None = None
    curseforge_api_key: str | None = None

    browser_use_primary_llm: str = "browser-use-llm"
    browser_use_fallback_llm: str = "browser-use-2.0"
    browser_use_planet_minecraft_skill_id: str | None = None
    browser_use_profile_id: str | None = None
    browser_use_task_timeout_seconds: int = 300

    max_plan_blocks: int = 8000
    app_host: str = "127.0.0.1"
    app_port: int = 8080

    laminar_api_key: str | None = None
    convex_url: str | None = None
    convex_access_key: str | None = None
    google_api_key: str | None = None
    anthropic_api_key: str | None = None
    anthropic_chat_model: str = "claude-sonnet-4-20250514"
    anthropic_chat_escalation_model: str = "claude-opus-4-1-20250805"
    anthropic_vision_model: str = "claude-sonnet-4-20250514"
    supermemory_api_key: str | None = None

    imagine_use_gemini_text_plan: bool = True

    allowed_download_exts: tuple[str, ...] = (".schem", ".litematic", ".schematic")


@lru_cache
def get_settings() -> Settings:
    return Settings()
