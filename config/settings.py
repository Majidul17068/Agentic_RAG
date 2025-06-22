"""
Configuration settings for SopAssist AI
"""

import os
from pathlib import Path
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent
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
    "lang": "eng",
    "config": "--psm 6 --oem 3",
    "timeout": 30
}

# PDF Processing Settings
PDF_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "supported_formats": [".pdf"]
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
    "base_url": "http://localhost:11434",
    "timeout": 60
}

# Free LLM Settings
FREE_LLM_CONFIG = {
    "groq": {
        "default_model": "llama3-8b-8192",
        "api_key_env": "GROQ_API_KEY",
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
    "port": 8000,
    "reload": True,
    "workers": 1
}

# Streamlit Settings
STREAMLIT_CONFIG = {
    "port": 8501,
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

# Policy Categories
POLICY_CATEGORIES = {
    "travel": ["ta", "da", "travel", "air", "ticket", "driver", "trip", "allowance"],
    "housing": ["house", "rent", "furniture", "cook", "accommodation"],
    "salary": ["salary", "increment", "bonus", "compensation", "pay"],
    "medical": ["medical", "health", "bill", "treatment"],
    "work_conditions": ["holiday", "overtime", "night", "shift", "attendance"],
    "allowances": ["allowance", "hardship", "location", "uniform", "mobile"],
    "employee_management": ["recruitment", "retirement", "termination", "notice"],
    "financial": ["financial", "budget", "expense", "claim", "reimbursement"],
    "farm_operations": ["farm", "hatchery", "production", "bio"],
    "general": ["policy", "circular", "notice", "order", "procedure"]
}

# Environment variables
def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with fallback to default."""
    return os.getenv(key, default)

# Override settings with environment variables if present
OLLAMA_CONFIG["base_url"] = get_env_var("OLLAMA_BASE_URL", OLLAMA_CONFIG["base_url"])
API_CONFIG["port"] = int(get_env_var("API_PORT", str(API_CONFIG["port"])))
STREAMLIT_CONFIG["port"] = int(get_env_var("STREAMLIT_PORT", str(STREAMLIT_CONFIG["port"])))

# LLM Provider Selection
DEFAULT_LLM_PROVIDER = get_env_var("LLM_PROVIDER", "groq")  # "groq", "huggingface", or "ollama" 