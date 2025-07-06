import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Google Gemini API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your_gemini_api_key_here")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost:5432/hr_assistant")
    
    # Vector Database Configuration (PostgreSQL with pgvector)
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "hr_assistant")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
    
    # Application Configuration
    APP_NAME = os.getenv("APP_NAME", "HR Knowledge Assistant")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Document Storage
    DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "./documents")
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    
    # Chunking Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Embedding Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # PostgreSQL Connection String
    @property
    def POSTGRES_URL(self):
        return os.getenv("DATABASE_URL")

# Create global config instance
config = Config() 