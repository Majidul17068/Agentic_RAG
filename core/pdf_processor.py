"""
PDF Processor for extracting text from policy PDF files
"""

import PyPDF2
import pdfplumber
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger
import sys
import os

# Add config to path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import FILE_CONFIG


class PDFProcessor:
    """Handles PDF processing of policy documents."""
    
    def __init__(self):
        """Initialize the PDF processor."""
        self.supported_formats = [".pdf"]
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file using multiple methods.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Try pdfplumber first (better for complex layouts)
            try:
                text = self._extract_with_pdfplumber(pdf_path)
                if text.strip():
                    logger.info(f"Successfully extracted {len(text)} characters using pdfplumber")
                    return self._clean_text(text)
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")
            
            # Fallback to PyPDF2
            try:
                text = self._extract_with_pypdf2(pdf_path)
                if text.strip():
                    logger.info(f"Successfully extracted {len(text)} characters using PyPDF2")
                    return self._clean_text(text)
            except Exception as e:
                logger.warning(f"PyPDF2 failed: {e}")
            
            logger.error(f"Failed to extract text from {pdf_path}")
            return ""
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return ""
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber."""
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return "\n".join(text_parts)
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2."""
        text_parts = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return "\n".join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing artifacts and normalizing.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\/\&\$\%\#\@\+\=]', '', text)
        
        # Fix common OCR issues
        text = text.replace('|', 'I')  # Common OCR mistake
        text = text.replace('0', 'O')  # In some contexts
        
        return text.strip()
    
    def categorize_policy(self, filename: str) -> Dict[str, str]:
        """
        Categorize policy based on filename.
        
        Args:
            filename: Name of the policy file
            
        Returns:
            Dictionary with category information
        """
        filename_lower = filename.lower()
        
        # Define policy categories and keywords
        categories = {
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
        
        # Find matching categories
        matched_categories = []
        for category, keywords in categories.items():
            if any(keyword in filename_lower for keyword in keywords):
                matched_categories.append(category)
        
        # Extract date if present
        date_match = re.search(r'(\d{4})', filename)
        year = date_match.group(1) if date_match else "Unknown"
        
        # Extract policy type from filename
        policy_type = self._extract_policy_type(filename)
        
        return {
            "categories": matched_categories if matched_categories else ["general"],
            "year": year,
            "policy_type": policy_type,
            "filename": filename
        }
    
    def _extract_policy_type(self, filename: str) -> str:
        """Extract policy type from filename."""
        # Remove file extension
        name = Path(filename).stem
        
        # Remove date patterns
        name = re.sub(r'\d{4}', '', name)
        name = re.sub(r'\(\w+,\s*\d{4}\)', '', name)
        
        # Clean up
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def process_policy_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Dict]:
        """
        Process all PDF files in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save extracted text
            
        Returns:
            Dictionary mapping filenames to extracted data
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        
        # Find all PDF files
        pdf_files = list(input_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in pdf_files:
            try:
                # Extract text
                text = self.extract_text_from_pdf(pdf_path)
                
                if text.strip():
                    # Categorize policy
                    categorization = self.categorize_policy(pdf_path.name)
                    
                    # Save extracted text
                    text_file = output_dir / f"{pdf_path.stem}.txt"
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    # Store results
                    results[pdf_path.name] = {
                        "text": text,
                        "categorization": categorization,
                        "text_file": str(text_file),
                        "file_size": pdf_path.stat().st_size,
                        "text_length": len(text)
                    }
                    
                    logger.info(f"Processed: {pdf_path.name} -> {categorization['categories']}")
                else:
                    logger.warning(f"No text extracted from {pdf_path.name}")
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
        
        logger.info(f"Successfully processed {len(results)} PDF files")
        return results
    
    def get_policy_summary(self, results: Dict[str, Dict]) -> Dict:
        """
        Generate summary of processed policies.
        
        Args:
            results: Results from process_policy_directory
            
        Returns:
            Summary statistics
        """
        total_files = len(results)
        total_text_length = sum(r["text_length"] for r in results.values())
        
        # Count by category
        category_counts = {}
        year_counts = {}
        
        for result in results.values():
            categories = result["categorization"]["categories"]
            year = result["categorization"]["year"]
            
            for category in categories:
                category_counts[category] = category_counts.get(category, 0) + 1
            
            year_counts[year] = year_counts.get(year, 0) + 1
        
        return {
            "total_files": total_files,
            "total_text_length": total_text_length,
            "average_text_length": total_text_length / total_files if total_files > 0 else 0,
            "category_distribution": category_counts,
            "year_distribution": year_counts
        }


def main():
    """Test the PDF processor with sample files."""
    processor = PDFProcessor()
    
    # Test with sample PDFs if they exist
    test_dir = Path("Policy file")
    output_dir = Path("data/processed_pdfs")
    
    if test_dir.exists() and any(test_dir.glob("*.pdf")):
        results = processor.process_policy_directory(test_dir, output_dir)
        
        # Generate summary
        summary = processor.get_policy_summary(results)
        
        print(f"\n📊 Processing Summary:")
        print(f"Total files: {summary['total_files']}")
        print(f"Total text length: {summary['total_text_length']:,} characters")
        print(f"Average text length: {summary['average_text_length']:,.0f} characters")
        
        print(f"\n📂 Category Distribution:")
        for category, count in summary['category_distribution'].items():
            print(f"  {category}: {count} files")
        
        print(f"\n📅 Year Distribution:")
        for year, count in summary['year_distribution'].items():
            print(f"  {year}: {count} files")
        
        # Show sample results
        print(f"\n📄 Sample Results:")
        for i, (filename, result) in enumerate(list(results.items())[:3]):
            print(f"\n{i+1}. {filename}")
            print(f"   Categories: {result['categorization']['categories']}")
            print(f"   Year: {result['categorization']['year']}")
            print(f"   Text length: {result['text_length']:,} characters")
            print(f"   Preview: {result['text'][:200]}...")
    else:
        print("No PDF files found in Policy file/ directory")
        print("Please add some PDF policy files and run again")


if __name__ == "__main__":
    main() 