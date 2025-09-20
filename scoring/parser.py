"""
Text extraction module for resumes and job descriptions.
Supports PDF and DOCX files with fallback mechanisms.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import re

# PDF processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# DOCX processing
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Text processing
try:
    import docx2txt
    DOCX2TXT_AVAILABLE = True
except ImportError:
    DOCX2TXT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentParser:
    """Main parser class for extracting text from various document formats."""
    
    def __init__(self):
        self.supported_formats = []
        if PYMUPDF_AVAILABLE or PDFPLUMBER_AVAILABLE:
            self.supported_formats.append('.pdf')
        if DOCX_AVAILABLE or DOCX2TXT_AVAILABLE:
            self.supported_formats.append('.docx')
    
    def extract_text(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "text": "",
                "metadata": {}
            }
        
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            return {
                "success": False,
                "error": f"Unsupported file format: {file_ext}",
                "text": "",
                "metadata": {}
            }
        
        try:
            if file_ext == '.pdf':
                return self._extract_pdf(file_path)
            elif file_ext == '.docx':
                return self._extract_docx(file_path)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file format: {file_ext}",
                    "text": "",
                    "metadata": {}
                }
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Extraction failed: {str(e)}",
                "text": "",
                "metadata": {}
            }
    
    def _extract_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF using PyMuPDF with pdfplumber fallback."""
        text = ""
        metadata = {"method": "none", "pages": 0}
        
        # Try PyMuPDF first
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(file_path)
                text_parts = []
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    page_text = page.get_text()
                    text_parts.append(page_text)
                
                text = "\n".join(text_parts)
                metadata = {
                    "method": "pymupdf",
                    "pages": doc.page_count,
                    "file_size": file_path.stat().st_size
                }
                doc.close()
                
                if text.strip():
                    return {
                        "success": True,
                        "text": self._clean_text(text),
                        "metadata": metadata,
                        "error": None
                    }
            except Exception as e:
                logger.warning(f"PyMuPDF failed for {file_path}: {str(e)}")
        
        # Fallback to pdfplumber
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(file_path) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    
                    text = "\n".join(text_parts)
                    metadata = {
                        "method": "pdfplumber",
                        "pages": len(pdf.pages),
                        "file_size": file_path.stat().st_size
                    }
                
                if text.strip():
                    return {
                        "success": True,
                        "text": self._clean_text(text),
                        "metadata": metadata,
                        "error": None
                    }
            except Exception as e:
                logger.warning(f"pdfplumber failed for {file_path}: {str(e)}")
        
        return {
            "success": False,
            "error": "All PDF extraction methods failed",
            "text": "",
            "metadata": metadata
        }
    
    def _extract_docx(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from DOCX using python-docx with docx2txt fallback."""
        text = ""
        metadata = {"method": "none"}
        
        # Try python-docx first
        if DOCX_AVAILABLE:
            try:
                doc = Document(file_path)
                text_parts = []
                for paragraph in doc.paragraphs:
                    text_parts.append(paragraph.text)
                
                text = "\n".join(text_parts)
                metadata = {
                    "method": "python-docx",
                    "file_size": file_path.stat().st_size
                }
                
                if text.strip():
                    return {
                        "success": True,
                        "text": self._clean_text(text),
                        "metadata": metadata,
                        "error": None
                    }
            except Exception as e:
                logger.warning(f"python-docx failed for {file_path}: {str(e)}")
        
        # Fallback to docx2txt
        if DOCX2TXT_AVAILABLE:
            try:
                text = docx2txt.process(str(file_path))
                metadata = {
                    "method": "docx2txt",
                    "file_size": file_path.stat().st_size
                }
                
                if text.strip():
                    return {
                        "success": True,
                        "text": self._clean_text(text),
                        "metadata": metadata,
                        "error": None
                    }
            except Exception as e:
                logger.warning(f"docx2txt failed for {file_path}: {str(e)}")
        
        return {
            "success": False,
            "error": "All DOCX extraction methods failed",
            "text": "",
            "metadata": metadata
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        text = re.sub(r'\f', '\n', text)  # Replace form feeds with newlines
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def save_extracted_text(self, text: str, original_filename: str, output_dir: str = "data/extracted_texts") -> str:
        """Save extracted text to a file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename based on original file
        base_name = Path(original_filename).stem
        txt_file = output_path / f"{base_name}.txt"
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return str(txt_file)

def extract_resume_text(file_path: str) -> Dict[str, Any]:
    """Convenience function to extract text from a resume."""
    parser = DocumentParser()
    return parser.extract_text(file_path)

def extract_jd_text(file_path: str) -> Dict[str, Any]:
    """Convenience function to extract text from a job description."""
    parser = DocumentParser()
    return parser.extract_text(file_path)

# Example usage
if __name__ == "__main__":
    parser = DocumentParser()
    print(f"Supported formats: {parser.supported_formats}")
    
    # Test with a sample file if available
    test_files = [
        "data/resumes/sample_resume.pdf",
        "data/jds/sample_jd.pdf"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nTesting {test_file}:")
            result = parser.extract_text(test_file)
            print(f"Success: {result['success']}")
            if result['success']:
                print(f"Text length: {len(result['text'])}")
                print(f"Method: {result['metadata'].get('method', 'unknown')}")
            else:
                print(f"Error: {result['error']}")
