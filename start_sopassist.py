#!/usr/bin/env python3
"""
SopAssist AI Startup Script
Comprehensive script to initialize and run the SopAssist AI system
"""

import sys
import subprocess
import time
import os
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import API_CONFIG, STREAMLIT_CONFIG


def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("🔍 Checking dependencies...")
    
    # Check Python version
    if sys.version_info < (3, 9):
        logger.error("❌ Python 3.9 or higher is required")
        return False
    
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        logger.info("✅ Tesseract OCR is available")
    except Exception as e:
        logger.error(f"❌ Tesseract OCR not found: {e}")
        logger.info("Please install Tesseract:")
        logger.info("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        logger.info("  macOS: brew install tesseract")
        return False
    
    # Check Ollama
    try:
        import ollama
        ollama.list()
        logger.info("✅ Ollama is available")
    except Exception as e:
        logger.error(f"❌ Ollama not found or not running: {e}")
        logger.info("Please install and start Ollama:")
        logger.info("  1. Install: curl -fsSL https://ollama.ai/install.sh | sh")
        logger.info("  2. Start: ollama serve")
        logger.info("  3. Pull model: ollama pull llama3")
        return False
    
    # Check Python packages
    required_packages = [
        "fastapi", "streamlit", "chromadb", "sentence_transformers", 
        "crewai", "pytesseract", "pillow", "opencv-python"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}")
    
    if missing_packages:
        logger.error("Missing Python packages. Please install with:")
        logger.error(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def setup_directories():
    """Create necessary directories."""
    logger.info("📁 Setting up directories...")
    
    directories = [
        "data/images",
        "data/processed", 
        "data/embeddings",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created: {directory}")


def add_sample_data():
    """Add sample policies to the system."""
    logger.info("📝 Adding sample policies...")
    
    try:
        from scripts.add_sample_policies import add_sample_policies
        add_sample_policies()
        logger.info("✅ Sample policies added")
        return True
    except Exception as e:
        logger.error(f"❌ Error adding sample policies: {e}")
        return False


def start_api():
    """Start the FastAPI backend."""
    logger.info("🚀 Starting API server...")
    
    api_script = Path("api/main.py")
    if not api_script.exists():
        logger.error(f"❌ API script not found: {api_script}")
        return None
    
    try:
        process = subprocess.Popen([
            sys.executable, str(api_script)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(3)
        
        if process.poll() is None:
            logger.info("✅ API server started")
            return process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ API server failed to start: {stderr.decode()}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error starting API: {e}")
        return None


def start_frontend():
    """Start the Streamlit frontend."""
    logger.info("🌐 Starting Streamlit frontend...")
    
    frontend_script = Path("frontend/app.py")
    if not frontend_script.exists():
        logger.error(f"❌ Frontend script not found: {frontend_script}")
        return None
    
    try:
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", str(frontend_script),
            "--server.port", str(STREAMLIT_CONFIG["port"]),
            "--server.address", STREAMLIT_CONFIG["host"]
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(5)
        
        if process.poll() is None:
            logger.info("✅ Streamlit frontend started")
            return process
        else:
            stdout, stderr = process.communicate()
            logger.error(f"❌ Frontend failed to start: {stderr.decode()}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error starting frontend: {e}")
        return None


def run_tests():
    """Run the test suite."""
    logger.info("🧪 Running tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ All tests passed")
            return True
        else:
            logger.error(f"❌ Tests failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running tests: {e}")
        return False


def show_menu():
    """Show the main menu."""
    print("\n" + "="*50)
    print("📋 SopAssist AI - Main Menu")
    print("="*50)
    print("1. Check Dependencies")
    print("2. Setup Directories")
    print("3. Add Sample Data")
    print("4. Run Tests")
    print("5. Start API Server")
    print("6. Start Frontend")
    print("7. Start Both (API + Frontend)")
    print("8. Process Images")
    print("9. Exit")
    print("="*50)


def main():
    """Main function."""
    logger.info("🚀 SopAssist AI Startup Script")
    
    while True:
        show_menu()
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == "1":
            if check_dependencies():
                logger.info("✅ All dependencies are satisfied")
            else:
                logger.error("❌ Some dependencies are missing")
        
        elif choice == "2":
            setup_directories()
        
        elif choice == "3":
            add_sample_data()
        
        elif choice == "4":
            run_tests()
        
        elif choice == "5":
            api_process = start_api()
            if api_process:
                logger.info(f"API running on http://{API_CONFIG['host']}:{API_CONFIG['port']}")
                logger.info("Press Ctrl+C to stop the API server")
                try:
                    api_process.wait()
                except KeyboardInterrupt:
                    logger.info("Stopping API server...")
                    api_process.terminate()
        
        elif choice == "6":
            frontend_process = start_frontend()
            if frontend_process:
                logger.info(f"Frontend running on http://{STREAMLIT_CONFIG['host']}:{STREAMLIT_CONFIG['port']}")
                logger.info("Press Ctrl+C to stop the frontend")
                try:
                    frontend_process.wait()
                except KeyboardInterrupt:
                    logger.info("Stopping frontend...")
                    frontend_process.terminate()
        
        elif choice == "7":
            logger.info("Starting both API and frontend...")
            api_process = start_api()
            if api_process:
                time.sleep(2)  # Wait for API to start
                frontend_process = start_frontend()
                if frontend_process:
                    logger.info(f"✅ System running!")
                    logger.info(f"API: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
                    logger.info(f"Frontend: http://{STREAMLIT_CONFIG['host']}:{STREAMLIT_CONFIG['port']}")
                    logger.info("Press Ctrl+C to stop both servers")
                    
                    try:
                        # Wait for either process to finish
                        while api_process.poll() is None and frontend_process.poll() is None:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        logger.info("Stopping servers...")
                        if api_process:
                            api_process.terminate()
                        if frontend_process:
                            frontend_process.terminate()
        
        elif choice == "8":
            logger.info("Processing images...")
            try:
                from scripts.process_images import process_images
                process_images()
            except Exception as e:
                logger.error(f"Error processing images: {e}")
        
        elif choice == "9":
            logger.info("👋 Goodbye!")
            break
        
        else:
            logger.warning("Invalid choice. Please enter 1-9.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1) 