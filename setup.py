#!/usr/bin/env python3
"""
Setup script for SopAssist AI
Creates necessary directories and initializes the project structure.
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure."""
    directories = [
        "api",
        "frontend", 
        "core",
        "agents",
        "data/images",
        "data/processed",
        "data/embeddings",
        "scripts",
        "tests",
        "config",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_init_files():
    """Create __init__.py files for Python packages."""
    init_dirs = ["api", "core", "agents", "tests"]
    
    for directory in init_dirs:
        init_file = Path(directory) / "__init__.py"
        init_file.touch(exist_ok=True)
        print(f"✓ Created __init__.py: {init_file}")

def check_dependencies():
    """Check if required system dependencies are installed."""
    print("\n🔍 Checking system dependencies...")
    
    # Check Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("✓ Tesseract OCR is available")
    except Exception as e:
        print("❌ Tesseract OCR not found. Please install it:")
        print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("   macOS: brew install tesseract")
        print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    
    # Check Ollama
    try:
        import ollama
        # Try to list models to check if Ollama is running
        ollama.list()
        print("✓ Ollama is available")
    except Exception as e:
        print("❌ Ollama not found or not running. Please:")
        print("   1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   2. Start Ollama: ollama serve")
        print("   3. Pull Llama3: ollama pull llama3")
        return False
    
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up SopAssist AI...")
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    create_directory_structure()
    
    # Create __init__.py files
    print("\n📄 Creating Python package files...")
    create_init_files()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    print("\n✅ Setup complete!")
    
    if deps_ok:
        print("\n🎉 All dependencies are ready!")
        print("\nNext steps:")
        print("1. Add your policy images to data/images/")
        print("2. Run: python scripts/process_images.py")
        print("3. Start the API: python api/main.py")
        print("4. Start the frontend: streamlit run frontend/app.py")
    else:
        print("\n⚠️  Please install missing dependencies before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main() 