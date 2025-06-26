#!/usr/bin/env python3
"""
Startup script for SopAssist AI with Bengali language support
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path
from loguru import logger

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.settings import validate_api_keys, print_config_summary, API_CONFIG, STREAMLIT_CONFIG
except ImportError as e:
    logger.error(f"Failed to import configuration: {e}")
    sys.exit(1)

class SopAssistStartup:
    """Startup manager for SopAssist AI."""
    
    def __init__(self):
        self.processes = []
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping all processes...")
        self.running = False
        self.stop_all_processes()
        sys.exit(0)
    
    def validate_environment(self):
        """Validate the environment and dependencies."""
        logger.info("🔍 Validating environment...")
        
        # Check Python version
        if sys.version_info < (3, 9):
            logger.error("❌ Python 3.9 or higher is required")
            return False
        
        logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        
        # Check required directories
        required_dirs = [
            "data",
            "data/images", 
            "data/processed",
            "data/embeddings",
            "data/processed_pdfs",
            "logs"
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Directory: {dir_path}")
        
        # Validate API keys
        logger.info("🔑 Validating API keys...")
        print_config_summary()
        
        if not validate_api_keys():
            logger.warning("⚠️ Some API keys are missing, but continuing...")
        
        # Check Tesseract installation
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ Tesseract OCR installed")
            else:
                logger.warning("⚠️ Tesseract OCR not found or not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("⚠️ Tesseract OCR not found. Install with: sudo apt-get install tesseract-ocr")
        
        # Check Bengali Tesseract support
        try:
            result = subprocess.run(['tesseract', '--list-langs'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'ben' in result.stdout:
                logger.info("✅ Bengali Tesseract support available")
            else:
                logger.warning("⚠️ Bengali Tesseract support not found. Install with: sudo apt-get install tesseract-ocr-ben")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("⚠️ Cannot check Bengali Tesseract support")
        
        return True
    
    def install_dependencies(self):
        """Install required Python dependencies."""
        logger.info("📦 Installing Python dependencies...")
        
        try:
            # Install requirements
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Dependencies installed successfully")
                return True
            else:
                logger.error(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Dependency installation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error installing dependencies: {e}")
            return False
    
    def start_backend(self):
        """Start the FastAPI backend server."""
        logger.info("🚀 Starting FastAPI backend...")
        
        try:
            process = subprocess.Popen([
                sys.executable, '-m', 'uvicorn', 
                'backend.main:app',
                '--host', API_CONFIG['host'],
                '--port', str(API_CONFIG['port']),
                '--reload' if API_CONFIG['reload'] else '--no-reload'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            self.processes.append(('backend', process))
            logger.info(f"✅ Backend started on http://{API_CONFIG['host']}:{API_CONFIG['port']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the Streamlit frontend."""
        logger.info("🎨 Starting Streamlit frontend...")
        
        try:
            process = subprocess.Popen([
                sys.executable, '-m', 'streamlit', 'run',
                'frontend/app.py',
                '--server.port', str(STREAMLIT_CONFIG['port']),
                '--server.address', STREAMLIT_CONFIG['host']
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            self.processes.append(('frontend', process))
            logger.info(f"✅ Frontend started on http://{STREAMLIT_CONFIG['host']}:{STREAMLIT_CONFIG['port']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start frontend: {e}")
            return False
    
    def wait_for_backend(self, timeout=60):
        """Wait for backend to be ready."""
        logger.info("⏳ Waiting for backend to be ready...")
        
        import requests
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://{API_CONFIG['host']}:{API_CONFIG['port']}/health", timeout=10)
                if response.status_code == 200:
                    logger.info("✅ Backend is ready")
                    return True
            except requests.exceptions.RequestException as e:
                logger.debug(f"Backend not ready yet: {e}")
                pass
            
            time.sleep(2)
        
        logger.warning(f"⚠️ Backend not ready within {timeout} seconds")
        logger.info("The backend may still be starting up. You can check manually at:")
        logger.info(f"  Health check: http://{API_CONFIG['host']}:{API_CONFIG['port']}/health")
        logger.info(f"  API docs: http://{API_CONFIG['host']}:{API_CONFIG['port']}/docs")
        return False
    
    def start_all_services(self):
        """Start all services."""
        logger.info("🚀 Starting SopAssist AI services...")
        
        # Start backend
        if not self.start_backend():
            logger.error("❌ Failed to start backend, stopping...")
            return False
        
        # Wait for backend to be ready
        if not self.wait_for_backend():
            logger.warning("⚠️ Backend may not be fully ready")
        
        # Start frontend
        if not self.start_frontend():
            logger.error("❌ Failed to start frontend")
            return False
        
        logger.info("🎉 All services started successfully!")
        logger.info(f"📱 Frontend: http://{STREAMLIT_CONFIG['host']}:{STREAMLIT_CONFIG['port']}")
        logger.info(f"🔧 Backend API: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
        logger.info(f"📚 API Docs: http://{API_CONFIG['host']}:{API_CONFIG['port']}/docs")
        
        return True
    
    def monitor_processes(self):
        """Monitor running processes."""
        while self.running:
            for name, process in self.processes:
                if process.poll() is not None:
                    logger.error(f"❌ {name} process has stopped unexpectedly")
                    self.running = False
                    break
            
            time.sleep(5)
    
    def stop_all_processes(self):
        """Stop all running processes."""
        logger.info("🛑 Stopping all processes...")
        
        for name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️ {name} didn't stop gracefully, forcing...")
                process.kill()
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
    
    def run(self):
        """Main run method."""
        logger.info("🎯 SopAssist AI Startup")
        logger.info("=" * 50)
        
        # Validate environment
        if not self.validate_environment():
            logger.error("❌ Environment validation failed")
            return False
        
        # Install dependencies if needed
        if not self.install_dependencies():
            logger.error("❌ Dependency installation failed")
            return False
        
        # Start services
        if not self.start_all_services():
            logger.error("❌ Failed to start services")
            return False
        
        # Monitor processes
        try:
            self.monitor_processes()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop_all_processes()
        
        return True


def main():
    """Main entry point."""
    # Setup logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/startup.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB"
    )
    
    # Run startup
    startup = SopAssistStartup()
    success = startup.run()
    
    if success:
        logger.info("✅ SopAssist AI started successfully")
        sys.exit(0)
    else:
        logger.error("❌ SopAssist AI failed to start")
        sys.exit(1)


if __name__ == "__main__":
    main() 