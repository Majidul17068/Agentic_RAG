#!/usr/bin/env python3
"""
Script to extract all PDF policies and save them to a single text file
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.pdf_processor import PDFProcessor
from config.settings import POLICY_DIR

def extract_all_policies_to_file():
    """
    Extract all PDF policies and save them to a single text file.
    Each policy will be named by its PDF filename.
    """
    # Initialize PDF processor
    pdf_processor = PDFProcessor()
    
    # Output file path
    output_file = Path("data/all_policies.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(POLICY_DIR.glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {POLICY_DIR}")
        return False
    
    # Limit to first 10 files for testing
    pdf_files = pdf_files[:10]
    logger.info(f"Processing first 10 PDF files out of {len(list(POLICY_DIR.glob('*.pdf')))} total files")
    
    extracted_count = 0
    failed_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pdf_file in pdf_files:
            try:
                logger.info(f"Extracting: {pdf_file.name}")
                
                # Extract text from PDF
                text = pdf_processor.extract_text_from_pdf(pdf_file)
                
                if text and text.strip():
                    # Write policy header with filename
                    f.write(f"\n{'='*80}\n")
                    f.write(f"POLICY: {pdf_file.name}\n")
                    f.write(f"FILENAME: {pdf_file.stem}\n")
                    f.write(f"{'='*80}\n\n")
                    
                    # Write the extracted text
                    f.write(text)
                    f.write("\n\n")
                    
                    extracted_count += 1
                    logger.info(f"✅ Extracted {pdf_file.name} ({len(text)} characters)")
                else:
                    logger.warning(f"⚠️ No text extracted from {pdf_file.name}")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"❌ Error extracting {pdf_file.name}: {e}")
                failed_count += 1
    
    logger.info(f"\n📊 Extraction Results:")
    logger.info(f"Total files: {len(pdf_files)}")
    logger.info(f"Successfully extracted: {extracted_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Output file: {output_file}")
    
    if extracted_count > 0:
        logger.info(f"✅ All policies extracted to: {output_file}")
        logger.info("You can now start the backend and frontend!")
        return True
    else:
        logger.error("❌ No policies were extracted successfully")
        return False

def main():
    """Main function."""
    logger.info("📄 PDF Policy Extraction Script")
    logger.info("=" * 50)
    
    # Check if policy folder exists
    if not POLICY_DIR.exists():
        logger.error(f"❌ Policy folder not found: {POLICY_DIR}")
        logger.info("Please create the 'Policy file' folder and add your PDF policy files.")
        return False
    
    # Extract all policies
    success = extract_all_policies_to_file()
    
    if success:
        logger.info("\n🎉 Ready to start backend and frontend!")
        logger.info("Run: python api/main.py")
        logger.info("Then: streamlit run frontend/app.py")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 