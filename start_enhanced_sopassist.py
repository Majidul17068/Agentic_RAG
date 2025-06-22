#!/usr/bin/env python3
"""
Enhanced SopAssist AI Startup Script
Processes PDF policies and starts the complete agentic RAG system
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path
from loguru import logger
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import POLICY_DIR, PROCESSED_PDFS_DIR
from core.pdf_processor import PDFProcessor
from scripts.process_policy_pdfs import process_policy_pdfs


class EnhancedSopAssist:
    """Enhanced SopAssist AI startup and management class."""
    
    def __init__(self):
        """Initialize the enhanced SopAssist system."""
        self.processes = []
        self.api_process = None
        self.frontend_process = None
        
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        logger.info("🔍 Checking system dependencies...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 9):
            logger.error("❌ Python 3.9 or higher is required")
            return False
        
        logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check required packages
        required_packages = [
            "fastapi", "uvicorn", "streamlit", "chromadb", "sentence_transformers",
            "PyPDF2", "pdfplumber", "groq", "transformers", "torch", "crewai"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                logger.info(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package} not found")
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Run: pip install -r requirements.txt")
            return False
        
        # Check for Tesseract (for OCR)
        try:
            result = subprocess.run(["tesseract", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Tesseract OCR")
            else:
                logger.warning("⚠️ Tesseract OCR not found (optional for PDF processing)")
        except FileNotFoundError:
            logger.warning("⚠️ Tesseract OCR not found (optional for PDF processing)")
        
        # Check for Ollama (optional)
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Ollama")
            else:
                logger.warning("⚠️ Ollama not found (optional)")
        except FileNotFoundError:
            logger.warning("⚠️ Ollama not found (optional)")
        
        logger.info("✅ All core dependencies are available")
        return True
    
    def setup_directories(self):
        """Setup required directories."""
        logger.info("📁 Setting up directories...")
        
        directories = [
            "data",
            "data/images", 
            "data/processed",
            "data/embeddings",
            "data/processed_pdfs",
            "logs",
            "config"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created {directory}/")
        
        logger.info("✅ All directories created")
    
    def process_pdf_policies(self, force_reprocess: bool = False):
        """Process PDF policy files."""
        logger.info("📄 Processing PDF policy files...")
        
        if not POLICY_DIR.exists():
            logger.warning(f"Policy directory not found: {POLICY_DIR}")
            logger.info("Please add PDF policy files to the 'Policy file/' directory")
            return False
        
        pdf_files = list(POLICY_DIR.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in Policy file/ directory")
            logger.info("Please add some PDF policy files and run again")
            return False
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Check if already processed
        if not force_reprocess and PROCESSED_PDFS_DIR.exists():
            processed_files = list(PROCESSED_PDFS_DIR.glob("*.txt"))
            if len(processed_files) >= len(pdf_files) * 0.8:  # 80% threshold
                logger.info("PDFs already processed. Use --force-reprocess to reprocess")
                return True
        
        try:
            # Process PDFs
            process_policy_pdfs(
                input_dir=POLICY_DIR,
                output_dir=PROCESSED_PDFS_DIR,
                collection_name="policy_documents"
            )
            
            logger.info("✅ PDF policy processing complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing PDF policies: {e}")
            return False
    
    def run_tests(self):
        """Run basic tests to ensure system is working."""
        logger.info("🧪 Running system tests...")
        
        try:
            # Test PDF processor
            from core.pdf_processor import PDFProcessor
            processor = PDFProcessor()
            logger.info("✅ PDF processor test passed")
            
            # Test vector store
            from core.vector_store import VectorStore
            vector_store = VectorStore("test_collection")
            logger.info("✅ Vector store test passed")
            
            # Test free LLM interface (if possible)
            try:
                from core.free_llm_interface import FreeLLMInterface
                # Try Hugging Face (most likely to work)
                llm = FreeLLMInterface("huggingface")
                logger.info("✅ Free LLM interface test passed")
            except Exception as e:
                logger.warning(f"⚠️ Free LLM interface test failed: {e}")
            
            logger.info("✅ All tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            return False
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the enhanced API server."""
        logger.info(f"🚀 Starting enhanced API server on {host}:{port}...")
        
        try:
            # Start the enhanced API
            from api.enhanced_main import EnhancedSopAssistAPI
            api = EnhancedSopAssistAPI()
            
            # Run in a separate process
            import uvicorn
            config = uvicorn.Config(
                api.app,
                host=host,
                port=port,
                reload=False,
                log_level="info"
            )
            server = uvicorn.Server(config)
            
            # Start server in background
            import threading
            api_thread = threading.Thread(target=server.run, daemon=True)
            api_thread.start()
            
            self.api_process = api_thread
            logger.info(f"✅ Enhanced API server started on http://{host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start API server: {e}")
            return False
    
    def start_frontend(self, port: int = 8501):
        """Start the Streamlit frontend."""
        logger.info(f"🎨 Starting Streamlit frontend on port {port}...")
        
        try:
            # Start Streamlit
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                "frontend/app.py",
                "--server.port", str(port),
                "--server.headless", "true"
            ]
            
            self.frontend_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for startup
            time.sleep(3)
            
            if self.frontend_process.poll() is None:
                logger.info(f"✅ Frontend started on http://localhost:{port}")
                return True
            else:
                logger.error("❌ Frontend failed to start")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {e}")
            return False
    
    def show_system_info(self):
        """Show system information and status."""
        logger.info("📊 System Information:")
        
        # Show policy statistics
        if POLICY_DIR.exists():
            pdf_files = list(POLICY_DIR.glob("*.pdf"))
            logger.info(f"📄 PDF Policy Files: {len(pdf_files)}")
            
            # Show categories
            if pdf_files:
                processor = PDFProcessor()
                categories = {}
                for pdf_file in pdf_files[:10]:  # Sample first 10
                    cat = processor.categorize_policy(pdf_file.name)
                    for category in cat["categories"]:
                        categories[category] = categories.get(category, 0) + 1
                
                logger.info("📂 Policy Categories:")
                for category, count in sorted(categories.items()):
                    logger.info(f"  {category}: {count} files")
        
        # Show vector store stats
        try:
            from core.vector_store import VectorStore
            vector_store = VectorStore("policy_documents")
            stats = vector_store.get_collection_stats()
            logger.info(f"🗄️ Vector Store: {stats.get('total_documents', 0)} documents")
        except:
            logger.info("🗄️ Vector Store: Not initialized")
        
        # Show available LLM providers
        llm_providers = []
        try:
            import groq
            llm_providers.append("Groq")
        except:
            pass
        
        try:
            import transformers
            llm_providers.append("Hugging Face")
        except:
            pass
        
        try:
            import ollama
            llm_providers.append("Ollama")
        except:
            pass
        
        logger.info(f"🤖 Available LLM Providers: {', '.join(llm_providers) if llm_providers else 'None'}")
    
    def show_menu(self):
        """Show interactive menu."""
        while True:
            print("\n" + "="*60)
            print("🎯 Enhanced SopAssist AI - Policy Assistant")
            print("="*60)
            print("1. 🔍 Check Dependencies")
            print("2. 📁 Setup Directories")
            print("3. 📄 Process PDF Policies")
            print("4. 🧪 Run Tests")
            print("5. 🚀 Start API Server")
            print("6. 🎨 Start Frontend")
            print("7. 🔄 Start Both (API + Frontend)")
            print("8. 📊 Show System Info")
            print("9. 🗑️ Clean and Reprocess PDFs")
            print("0. ❌ Exit")
            print("="*60)
            
            choice = input("Enter your choice (0-9): ").strip()
            
            if choice == "1":
                self.check_dependencies()
            elif choice == "2":
                self.setup_directories()
            elif choice == "3":
                self.process_pdf_policies()
            elif choice == "4":
                self.run_tests()
            elif choice == "5":
                self.start_api_server()
            elif choice == "6":
                self.start_frontend()
            elif choice == "7":
                if self.start_api_server():
                    time.sleep(2)
                    self.start_frontend()
            elif choice == "8":
                self.show_system_info()
            elif choice == "9":
                self.process_pdf_policies(force_reprocess=True)
            elif choice == "0":
                logger.info("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
            
            input("\nPress Enter to continue...")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Enhanced SopAssist AI Startup")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies")
    parser.add_argument("--setup", action="store_true", help="Setup directories")
    parser.add_argument("--process-pdfs", action="store_true", help="Process PDF policies")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocess PDFs")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--start-api", action="store_true", help="Start API server")
    parser.add_argument("--start-frontend", action="store_true", help="Start frontend")
    parser.add_argument("--start-all", action="store_true", help="Start both API and frontend")
    parser.add_argument("--info", action="store_true", help="Show system info")
    parser.add_argument("--interactive", action="store_true", help="Start interactive menu")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    sopassist = EnhancedSopAssist()
    
    try:
        if args.check_deps:
            sopassist.check_dependencies()
        elif args.setup:
            sopassist.setup_directories()
        elif args.process_pdfs:
            sopassist.process_pdf_policies(args.force_reprocess)
        elif args.test:
            sopassist.run_tests()
        elif args.start_api:
            sopassist.start_api_server()
        elif args.start_frontend:
            sopassist.start_frontend()
        elif args.start_all:
            if sopassist.start_api_server():
                time.sleep(2)
                sopassist.start_frontend()
        elif args.info:
            sopassist.show_system_info()
        elif args.interactive:
            sopassist.show_menu()
        else:
            # Default: show interactive menu
            sopassist.show_menu()
            
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 