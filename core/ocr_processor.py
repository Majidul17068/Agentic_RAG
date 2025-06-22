"""
OCR Processor for extracting text from policy images
"""

import pytesseract
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
import sys
import os

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import OCR_CONFIG, FILE_CONFIG


class OCRProcessor:
    """Handles OCR processing of policy images."""
    
    def __init__(self):
        """Initialize the OCR processor."""
        self.config = OCR_CONFIG
        self.supported_formats = FILE_CONFIG["supported_formats"]
        
        # Verify Tesseract is available
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR initialized successfully")
        except Exception as e:
            logger.error(f"Tesseract OCR not available: {e}")
            raise
    
    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text(self, image_path: Path, preprocess: bool = True) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to preprocess the image
            
        Returns:
            Extracted text as string
        """
        try:
            logger.info(f"Processing image: {image_path}")
            
            if preprocess:
                # Use OpenCV preprocessing
                processed_image = self.preprocess_image(image_path)
                text = pytesseract.image_to_string(
                    processed_image,
                    lang=self.config["lang"],
                    config=self.config["config"],
                    timeout=self.config["timeout"]
                )
            else:
                # Use PIL directly
                image = Image.open(image_path)
                text = pytesseract.image_to_string(
                    image,
                    lang=self.config["lang"],
                    config=self.config["config"],
                    timeout=self.config["timeout"]
                )
            
            # Clean up text
            text = self._clean_text(text)
            
            logger.info(f"Successfully extracted {len(text)} characters from {image_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and normalizing.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove common OCR artifacts
        text = text.replace("|", "I")  # Common OCR mistake
        text = text.replace("0", "O")  # In some contexts
        
        return text.strip()
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, str]:
        """
        Process all images in a directory and save extracted text.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory to save extracted text
            
        Returns:
            Dictionary mapping image names to extracted text
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        
        # Find all supported image files
        image_files = []
        for format_ext in self.supported_formats:
            image_files.extend(input_dir.glob(f"*{format_ext}"))
            image_files.extend(input_dir.glob(f"*{format_ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        for image_path in image_files:
            try:
                # Extract text
                text = self.extract_text(image_path)
                
                if text.strip():
                    # Save to file
                    output_file = output_dir / f"{image_path.stem}.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    results[image_path.name] = text
                    logger.info(f"Saved text for {image_path.name}")
                else:
                    logger.warning(f"No text extracted from {image_path.name}")
                    
            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")
        
        logger.info(f"Successfully processed {len(results)} images")
        return results
    
    def get_text_confidence(self, image_path: Path) -> Dict[str, float]:
        """
        Get OCR confidence scores for extracted text.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with confidence scores
        """
        try:
            image = Image.open(image_path)
            data = pytesseract.image_to_data(
                image,
                lang=self.config["lang"],
                config=self.config["config"],
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "average_confidence": avg_confidence,
                "word_count": len(confidences),
                "high_confidence_words": len([c for c in confidences if c > 80])
            }
            
        except Exception as e:
            logger.error(f"Error getting confidence for {image_path}: {e}")
            return {"average_confidence": 0, "word_count": 0, "high_confidence_words": 0}


def main():
    """Test the OCR processor with sample images."""
    processor = OCRProcessor()
    
    # Test with sample images if they exist
    test_dir = Path("data/images")
    output_dir = Path("data/processed")
    
    if test_dir.exists() and any(test_dir.glob("*")):
        results = processor.process_directory(test_dir, output_dir)
        print(f"Processed {len(results)} images")
        for image_name, text in results.items():
            print(f"\n{image_name}:")
            print(f"Text length: {len(text)} characters")
            print(f"Preview: {text[:200]}...")
    else:
        print("No test images found in data/images/")
        print("Please add some images and run again")


if __name__ == "__main__":
    main() 