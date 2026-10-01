from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str
    postgres_user: str = "paperlens"
    postgres_password: str = "paperlens"
    postgres_db: str = "paperlens"
    postgres_port: int = 5432
    llm_model: str = "qwen2.5:7b"
    embedding_model: str = "BAAI/bge-m3"
    llm_api_key: str = "ollama"
    llm_base_url: str = "http://localhost:11434/v1"


settings = Settings()
