import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API keys and credentials
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")

    # Database settings (if needed later)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./interactive_story.db")

    # Vector database settings
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chroma")  # Options: chroma, pinecone
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

    # Application settings
    UPLOAD_DIRECTORY: str = os.getenv("UPLOAD_DIRECTORY", "./uploads")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))

    # LLM settings (Updated for Groq)
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = os.getenv("LLM_MODEL", "mixtral-8x7b-32768")  # Free Groq model
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.7))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", 1000))

    class Config:
        env_file = ".env"
        case_sensitive = True

# Create a global settings instance
settings_obj = Settings()
