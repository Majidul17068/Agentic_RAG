"""
Configuration settings for SopAssist AI with Bengali language support
"""

import os
from pathlib import Path
from typing import Optional

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# API Configuration
API_PORT = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# Ollama Configuration (if using local Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Database Configuration
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "data" / "embeddings"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(BASE_DIR / "logs" / "app.log"))

# File Processing Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB in bytes
SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", "jpg,jpeg,png,tiff,bmp,pdf").split(",")

# Search Configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# Bengali Language Support
BENGALI_DETECTION_THRESHOLD = float(os.getenv("BENGALI_DETECTION_THRESHOLD", "0.1"))
BENGALI_OCR_LANG = os.getenv("BENGALI_OCR_LANG", "ben+eng")

# Base paths
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LOGS_DIR = BASE_DIR / "logs"
POLICY_DIR = BASE_DIR / "Policy file"
PROCESSED_PDFS_DIR = DATA_DIR / "processed_pdfs"

# Ensure directories exist
for directory in [DATA_DIR, IMAGES_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, LOGS_DIR, PROCESSED_PDFS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# OCR Settings
OCR_CONFIG = {
    "lang": "eng+ben",  # English + Bengali
    "config": "--psm 6 --oem 3",
    "timeout": 30,
    "dpi": 300
}

# PDF Processing Settings
PDF_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "supported_formats": [".pdf"],
    "ocr_fallback": True,  # Use OCR if text extraction fails
    "bengali_support": True  # Enable Bengali language support
}

# Embedding Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 32

# Vector Database Settings
CHROMA_SETTINGS = {
    "persist_directory": str(EMBEDDINGS_DIR),
    "anonymized_telemetry": False
}

# Ollama Settings
OLLAMA_CONFIG = {
    "model": "llama3",
    "base_url": OLLAMA_BASE_URL,
    "timeout": 60
}

# Free LLM Settings
FREE_LLM_CONFIG = {
    "groq": {
        "default_model": "llama3-8b-8192",
        "api_key": GROQ_API_KEY,
        "timeout": 30,
        "max_tokens": 1000
    },
    "huggingface": {
        "default_model": "microsoft/DialoGPT-medium",
        "device": "auto",  # "cpu", "cuda", or "auto"
        "max_length": 512,
        "temperature": 0.7,
        "load_in_8bit": False,
        "load_in_4bit": False
    }
}

# API Settings
API_CONFIG = {
    "host": "0.0.0.0",
    "port": API_PORT,
    "reload": True,
    "workers": 1
}

# Streamlit Settings
STREAMLIT_CONFIG = {
    "port": STREAMLIT_PORT,
    "host": "localhost"
}

# CrewAI Settings
CREWAI_CONFIG = {
    "max_iterations": 3,
    "verbose": True
}

# Logging Settings
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    "file": str(LOGS_DIR / "app.log"),
    "error_file": str(LOGS_DIR / "error.log")
}

# File Processing Settings
FILE_CONFIG = {
    "supported_formats": [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".pdf"],
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "image_quality": 85
}

# Search Settings
SEARCH_CONFIG = {
    "top_k": 5,
    "similarity_threshold": 0.7,
    "max_context_length": 2000
}

# Policy Categories (English + Bengali)
POLICY_CATEGORIES = {
    "travel": ["ta", "da", "travel", "air", "ticket", "driver", "trip", "allowance", "ভ্রমণ", "টিকিট", "ভ্রমণ ভাতা"],
    "housing": ["house", "rent", "furniture", "cook", "accommodation", "বাড়ি", "ভাড়া", "ফার্নিচার", "রান্না"],
    "salary": ["salary", "increment", "bonus", "compensation", "pay", "বেতন", "বোনাস", "বৃদ্ধি", "মজুরি"],
    "medical": ["medical", "health", "bill", "treatment", "চিকিৎসা", "স্বাস্থ্য", "বিল", "চিকিৎসা বিল"],
    "work_conditions": ["holiday", "overtime", "night", "shift", "attendance", "ছুটি", "অতিরিক্ত", "রাত", "শিফট"],
    "allowances": ["allowance", "hardship", "location", "uniform", "mobile", "ভাতা", "কষ্ট", "ইউনিফর্ম", "মোবাইল"],
    "employee_management": ["recruitment", "retirement", "termination", "notice", "নিয়োগ", "অবসর", "বরখাস্ত", "নোটিশ"],
    "financial": ["financial", "budget", "expense", "claim", "reimbursement", "আর্থিক", "বাজেট", "খরচ", "দাবি"],
    "farm_operations": ["farm", "hatchery", "production", "bio", "খামার", "হ্যাচারি", "উৎপাদন", "জৈব"],
    "general": ["policy", "circular", "notice", "order", "procedure", "নীতি", "পরিপত্র", "নোটিশ", "আদেশ"]
}

# Language Support Settings
LANGUAGE_CONFIG = {
    "supported_languages": ["english", "bengali"],
    "default_language": "english",
    "bengali_detection_threshold": 0.1,  # 10% Bengali characters to consider Bengali
    "bengali_ocr_lang": "ben+eng",  # Bengali + English OCR
    "english_ocr_lang": "eng"
}

# Environment variables
def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with fallback to default."""
    return os.getenv(key, default)

# LLM Provider Selection
DEFAULT_LLM_PROVIDER = LLM_PROVIDER

# Validation functions
def validate_api_keys():
    """Validate that required API keys are set."""
    missing_keys = []
    
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")
    
    if missing_keys:
        print(f"⚠️ Missing required API keys: {', '.join(missing_keys)}")
        print("Please set these environment variables or update your .env file.")
        return False
    
    return True

def print_config_summary():
    """Print a summary of the current configuration."""
    print("🔧 SopAssist AI Configuration Summary:")
    print(f"   LLM Provider: {LLM_PROVIDER}")
    print(f"   API Port: {API_PORT}")
    print(f"   Streamlit Port: {STREAMLIT_PORT}")
    print(f"   Groq API Key: {'✅ Set' if GROQ_API_KEY else '❌ Missing'}")
    print(f"   Ollama URL: {OLLAMA_BASE_URL}")
    print(f"   Log Level: {LOG_LEVEL}")
    print(f"   Bengali Support: ✅ Enabled") 