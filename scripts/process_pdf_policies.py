#!/usr/bin/env python3
"""
Script to process PDF policy files into the vector store with Bengali support
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.pdf_processor import PDFProcessor
from core.vector_store import VectorStore
from config.settings import POLICY_DIR, PROCESSED_PDFS_DIR

def process_policy_folder(folder_path: Path) -> Dict[str, Any]:
    """
    Process all PDF files in a policy folder.
    
    Args:
        folder_path: Path to the folder containing PDF policy files
        
    Returns:
        Dictionary with processing results
    """
    results = {
        "total_files": 0,
        "processed_files": 0,
        "failed_files": 0,
        "total_chunks": 0,
        "categories": {},
        "languages": {},
        "errors": []
    }
    
    # Initialize processors
    pdf_processor = PDFProcessor()
    vector_store = VectorStore()
    
    # Get all PDF files
    pdf_files = list(folder_path.glob("*.pdf"))
    results["total_files"] = len(pdf_files)
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {folder_path}")
        return results
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")
            
            # Extract text from PDF
            text = pdf_processor.extract_text_from_pdf(pdf_file)
            
            if not text or not text.strip():
                logger.warning(f"No text extracted from {pdf_file.name}")
                results["failed_files"] += 1
                results["errors"].append(f"No text extracted from {pdf_file.name}")
                continue
            
            # Detect language
            language = pdf_processor.detect_language(text)
            # Categorize policy
            categorization = pdf_processor.categorize_policy(pdf_file.name)
            categorization["language"] = language
            
            # Store in vector database
            metadata = {
                "source": pdf_file.name,
                "type": "pdf",
                "file_id": f"pdf_{pdf_file.stem}_{hash(text)}",
                "category": categorization.get('categories', ['general']),
                "language": language,
                "file_path": str(pdf_file),
                "processing_date": str(Path(__file__).parent.parent / "data" / "processed_pdfs" / pdf_file.name)
            }
            chunks = vector_store.add_texts([text], [metadata])
            
            # Update results
            results["processed_files"] += 1
            results["total_chunks"] += len(chunks)
            
            # Track categories
            category = categorization.get('categories', ['general'])[0]
            results["categories"][category] = results["categories"].get(category, 0) + 1
            
            # Track languages
            results["languages"][language] = results["languages"].get(language, 0) + 1
            
            logger.info(f"✅ Processed {pdf_file.name} - Category: {category}, Language: {language}, Chunks: {len(chunks)}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_file.name}: {e}")
            results["failed_files"] += 1
            results["errors"].append(f"Error processing {pdf_file.name}: {str(e)}")
    
    return results

def main():
    """Main function to process policy PDFs."""
    # Setup logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("📄 PDF Policy Processing Script")
    logger.info("=" * 50)
    
    # Check if policy folder exists
    if not POLICY_DIR.exists():
        logger.error(f"❌ Policy folder not found: {POLICY_DIR}")
        logger.info("Please create the 'Policy file' folder and add your PDF policy files.")
        return False
    
    # Process the policy folder
    results = process_policy_folder(POLICY_DIR)
    
    # Print results
    logger.info("\n📊 Processing Results:")
    logger.info("=" * 30)
    logger.info(f"Total files: {results['total_files']}")
    logger.info(f"Processed: {results['processed_files']}")
    logger.info(f"Failed: {results['failed_files']}")
    logger.info(f"Total chunks: {results['total_chunks']}")
    
    if results['categories']:
        logger.info("\n📂 Categories:")
        for category, count in results['categories'].items():
            logger.info(f"  {category}: {count} files")
    
    if results['languages']:
        logger.info("\n🌐 Languages:")
        for language, count in results['languages'].items():
            logger.info(f"  {language}: {count} files")
    
    if results['errors']:
        logger.info("\n❌ Errors:")
        for error in results['errors']:
            logger.error(f"  {error}")
    
    # Summary
    if results['processed_files'] > 0:
        logger.info(f"\n✅ Successfully processed {results['processed_files']} PDF files")
        logger.info(f"📚 Added {results['total_chunks']} text chunks to vector database")
        logger.info("🎉 You can now query the policies using the SopAssist AI interface!")
        return True
    else:
        logger.error("❌ No files were processed successfully")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 