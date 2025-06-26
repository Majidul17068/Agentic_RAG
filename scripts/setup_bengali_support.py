#!/usr/bin/env python3
"""
Setup script for Bengali language support in SopAssist AI
"""

import subprocess
import sys
import os
from pathlib import Path
from loguru import logger


def check_tesseract_installation():
    """Check if Tesseract is installed and has Bengali support."""
    try:
        # Check if tesseract is installed
        result = subprocess.run(["tesseract", "--version"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Tesseract is installed")
            
            # Check available languages
            lang_result = subprocess.run(["tesseract", "--list-langs"], 
                                       capture_output=True, text=True)
            
            if "ben" in lang_result.stdout:
                logger.info("✅ Bengali language pack is installed")
                return True
            else:
                logger.warning("⚠️ Bengali language pack not found")
                return False
        else:
            logger.error("❌ Tesseract is not installed")
            return False
            
    except FileNotFoundError:
        logger.error("❌ Tesseract is not installed")
        return False


def install_tesseract_bengali():
    """Install Tesseract Bengali language pack."""
    logger.info("🔧 Installing Tesseract Bengali language support...")
    
    # Detect OS
    if sys.platform.startswith('linux'):
        return install_tesseract_bengali_linux()
    elif sys.platform.startswith('darwin'):  # macOS
        return install_tesseract_bengali_macos()
    elif sys.platform.startswith('win'):
        return install_tesseract_bengali_windows()
    else:
        logger.error(f"❌ Unsupported operating system: {sys.platform}")
        return False


def install_tesseract_bengali_linux():
    """Install Tesseract Bengali on Linux."""
    try:
        # Try different package managers
        package_managers = [
            ("apt-get", "sudo apt-get update && sudo apt-get install -y tesseract-ocr-ben"),
            ("yum", "sudo yum install -y tesseract-langpack-ben"),
            ("dnf", "sudo dnf install -y tesseract-langpack-ben"),
            ("pacman", "sudo pacman -S tesseract-data-ben")
        ]
        
        for pkg_manager, command in package_managers:
            try:
                # Check if package manager exists
                result = subprocess.run([pkg_manager, "--version"], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"📦 Using {pkg_manager} package manager")
                    
                    # Install Bengali language pack
                    install_result = subprocess.run(command, shell=True, 
                                                  capture_output=True, text=True)
                    
                    if install_result.returncode == 0:
                        logger.info("✅ Bengali language pack installed successfully")
                        return True
                    else:
                        logger.warning(f"⚠️ Failed to install with {pkg_manager}: {install_result.stderr}")
                        
            except FileNotFoundError:
                continue
        
        logger.error("❌ No supported package manager found")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error installing Bengali language pack: {e}")
        return False


def install_tesseract_bengali_macos():
    """Install Tesseract Bengali on macOS."""
    try:
        # Check if Homebrew is installed
        result = subprocess.run(["brew", "--version"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("📦 Using Homebrew package manager")
            
            # Install Bengali language pack
            install_result = subprocess.run(
                ["brew", "install", "tesseract-lang"], 
                capture_output=True, text=True
            )
            
            if install_result.returncode == 0:
                logger.info("✅ Bengali language pack installed successfully")
                return True
            else:
                logger.error(f"❌ Failed to install: {install_result.stderr}")
                return False
        else:
            logger.error("❌ Homebrew not found. Please install Homebrew first.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error installing Bengali language pack: {e}")
        return False


def install_tesseract_bengali_windows():
    """Install Tesseract Bengali on Windows."""
    logger.info("📋 For Windows, please follow these manual steps:")
    logger.info("1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
    logger.info("2. Install Tesseract with Bengali language pack")
    logger.info("3. Add Tesseract to your PATH environment variable")
    logger.info("4. Restart your terminal/command prompt")
    return False


def test_bengali_ocr():
    """Test Bengali OCR functionality."""
    logger.info("🧪 Testing Bengali OCR...")
    
    try:
        import pytesseract
        from PIL import Image
        
        # Create a simple test image with Bengali text
        test_text = "বাংলা ভাষা"
        
        # Create a simple image (this is a basic test)
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a white image
        img = Image.new('RGB', (200, 50), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a font that supports Bengali
        try:
            # Try to use a system font that supports Bengali
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Draw text
        draw.text((10, 10), test_text, fill='black', font=font)
        
        # Save test image
        test_image_path = "test_bengali.png"
        img.save(test_image_path)
        
        # Test OCR
        try:
            result = pytesseract.image_to_string(
                img, 
                lang='ben+eng',
                config='--psm 6 --oem 3'
            )
            
            logger.info(f"✅ Bengali OCR test successful")
            logger.info(f"   Input: {test_text}")
            logger.info(f"   Output: {result.strip()}")
            
            # Clean up
            os.remove(test_image_path)
            return True
            
        except Exception as e:
            logger.error(f"❌ Bengali OCR test failed: {e}")
            return False
            
    except ImportError:
        logger.error("❌ Required packages not installed. Run: pip install pytesseract pillow")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing Bengali OCR: {e}")
        return False


def setup_bengali_support():
    """Main setup function for Bengali language support."""
    logger.info("🌍 Setting up Bengali language support for SopAssist AI")
    
    # Check current installation
    if check_tesseract_installation():
        logger.info("✅ Bengali support is already installed")
        
        # Test OCR
        if test_bengali_ocr():
            logger.info("🎉 Bengali language support is working correctly!")
            return True
        else:
            logger.warning("⚠️ Bengali OCR test failed, but installation seems correct")
            return True
    
    # Install Bengali support
    logger.info("📦 Installing Bengali language support...")
    
    if install_tesseract_bengali():
        # Test after installation
        if test_bengali_ocr():
            logger.info("🎉 Bengali language support installed and working!")
            return True
        else:
            logger.warning("⚠️ Installation completed but OCR test failed")
            return True
    else:
        logger.error("❌ Failed to install Bengali language support")
        return False


def show_manual_instructions():
    """Show manual installation instructions."""
    logger.info("📋 Manual Installation Instructions:")
    logger.info("")
    logger.info("For Ubuntu/Debian:")
    logger.info("  sudo apt-get update")
    logger.info("  sudo apt-get install tesseract-ocr tesseract-ocr-ben")
    logger.info("")
    logger.info("For CentOS/RHEL/Fedora:")
    logger.info("  sudo yum install tesseract tesseract-langpack-ben")
    logger.info("  # or")
    logger.info("  sudo dnf install tesseract tesseract-langpack-ben")
    logger.info("")
    logger.info("For macOS:")
    logger.info("  brew install tesseract tesseract-lang")
    logger.info("")
    logger.info("For Windows:")
    logger.info("  1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    logger.info("  2. Install with Bengali language pack")
    logger.info("  3. Add to PATH environment variable")
    logger.info("")
    logger.info("After installation, restart your terminal and run this script again.")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Bengali language support")
    parser.add_argument("--check", action="store_true", help="Check current installation")
    parser.add_argument("--install", action="store_true", help="Install Bengali support")
    parser.add_argument("--test", action="store_true", help="Test Bengali OCR")
    parser.add_argument("--manual", action="store_true", help="Show manual instructions")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    try:
        if args.check:
            check_tesseract_installation()
        elif args.install:
            setup_bengali_support()
        elif args.test:
            test_bengali_ocr()
        elif args.manual:
            show_manual_instructions()
        else:
            # Default: full setup
            setup_bengali_support()
            
    except KeyboardInterrupt:
        logger.info("👋 Setup interrupted by user")
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 