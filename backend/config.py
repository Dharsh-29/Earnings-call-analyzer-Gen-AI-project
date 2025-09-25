# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    
    # Model Configuration
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
    
    # Vector Database Configuration
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1536"))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    
    # File Paths
    DEMO_PDF_PATH = os.getenv(
        "DEMO_PDF_PATH", 
        r"C:\Users\Dharsh\Documents\earningscall_analyzer\backend\data\demo_transcript\Q2FY24_LaurusLabs_EarningsCallTranscript.pdf"
    )
    
    # Application Settings
    APP_DEBUG = os.getenv("APP_DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if not os.path.exists(cls.DEMO_PDF_PATH):
            print(f"Warning: Demo PDF not found at {cls.DEMO_PDF_PATH}")
        
        return True

# Create a global config instance
config = Config()