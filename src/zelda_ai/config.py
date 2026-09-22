from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ZELDA_", env_file=".env", extra="ignore")

    data_dir: Path = Path(".local")
    bridge_port: int = Field(default=8766, ge=1024, le=65535)
    bridge_token: SecretStr = SecretStr("")
    codex_command: str = "codex"
    openrouter_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="OPENROUTER_API_KEY")
    decision_timeout_s: float = Field(default=180, ge=5, le=600)
    database_url: str | None = None
    allow_simulator: bool = False

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir.resolve() / 'zelda.sqlite3'}"

    @property
    def codex_home(self) -> Path:
        return self.data_dir.resolve() / "codex"
