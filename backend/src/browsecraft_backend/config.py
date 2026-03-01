from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 8080

    laminar_api_key: str | None = None
    convex_url: str | None = None
    convex_access_key: str | None = None
    anthropic_api_key: str | None = None
    anthropic_chat_model: str = "claude-sonnet-4-6"
    supermemory_api_key: str | None = None
    browser_use_api_key: str | None = None
    browser_use_llm: str = "browser-use-llm"
    browser_use_planet_minecraft_skill_id: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
