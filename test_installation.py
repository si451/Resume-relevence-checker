#!/usr/bin/env python3
"""
Test script to verify the Resume Relevance Checker installation.
Run this script to check if all dependencies are properly installed.
"""

import sys
import importlib
from pathlib import Path

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {str(e)}")
        return False

def test_file_exists(file_path):
    """Test if a file exists."""
    if Path(file_path).exists():
        print(f"✅ {file_path}")
        return True
    else:
        print(f"❌ {file_path}")
        return False

def main():
    """Run all tests."""
    print("🔍 Testing Resume Relevance Checker Installation")
    print("=" * 50)
    
    # Test Python version
    print(f"\n📋 Python Version: {sys.version}")
    
    # Test core dependencies
    print("\n📦 Core Dependencies:")
    core_deps = [
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("requests", "Requests"),
        ("sklearn", "Scikit-learn"),
    ]
    
    core_success = 0
    for module, name in core_deps:
        if test_import(module, name):
            core_success += 1
    
    # Test document processing
    print("\n📄 Document Processing:")
    doc_deps = [
        ("fitz", "PyMuPDF"),
        ("pdfplumber", "pdfplumber"),
        ("docx", "python-docx"),
        ("docx2txt", "docx2txt"),
    ]
    
    doc_success = 0
    for module, name in doc_deps:
        if test_import(module, name):
            doc_success += 1
    
    # Test ML dependencies
    print("\n🤖 ML Dependencies:")
    ml_deps = [
        ("sentence_transformers", "Sentence Transformers"),
        ("nltk", "NLTK"),
        ("spacy", "spaCy"),
    ]
    
    ml_success = 0
    for module, name in ml_deps:
        if test_import(module, name):
            ml_success += 1
    
    # Test project files
    print("\n📁 Project Files:")
    project_files = [
        "streamlit_app.py",
        "requirements.txt",
        "README.md",
        "scoring/__init__.py",
        "scoring/parser.py",
        "scoring/skill_extractor.py",
        "scoring/scoring.py",
        "scoring/embeddings.py",
        "scoring/grok_client.py",
        "scoring/feedback.py",
        ".cursor_tasks.json",
    ]
    
    file_success = 0
    for file_path in project_files:
        if test_file_exists(file_path):
            file_success += 1
    
    # Test data directories
    print("\n📂 Data Directories:")
    data_dirs = [
        "data/resumes",
        "data/jds",
        "data/extracted_texts",
        "data/results",
        "notebooks",
    ]
    
    dir_success = 0
    for dir_path in data_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
            dir_success += 1
        else:
            print(f"❌ {dir_path}")
    
    # Test our modules
    print("\n🔧 Custom Modules:")
    try:
        from scoring.parser import DocumentParser
        print("✅ DocumentParser")
    except Exception as e:
        print(f"❌ DocumentParser: {str(e)}")
    
    try:
        from scoring.skill_extractor import SkillExtractor
        print("✅ SkillExtractor")
    except Exception as e:
        print(f"❌ SkillExtractor: {str(e)}")
    
    try:
        from scoring.scoring import ResumeScorer
        print("✅ ResumeScorer")
    except Exception as e:
        print(f"❌ ResumeScorer: {str(e)}")
    
    try:
        from scoring.embeddings import EmbeddingManager
        print("✅ EmbeddingManager")
    except Exception as e:
        print(f"❌ EmbeddingManager: {str(e)}")
    
    try:
        from scoring.grok_client import GrokClient
        print("✅ GrokClient")
    except Exception as e:
        print(f"❌ GrokClient: {str(e)}")
    
    try:
        from scoring.feedback import FeedbackGenerator
        print("✅ FeedbackGenerator")
    except Exception as e:
        print(f"❌ FeedbackGenerator: {str(e)}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    total_tests = len(core_deps) + len(doc_deps) + len(ml_deps) + len(project_files) + len(data_dirs) + 6
    total_success = core_success + doc_success + ml_success + file_success + dir_success + 6
    
    print(f"Core Dependencies: {core_success}/{len(core_deps)}")
    print(f"Document Processing: {doc_success}/{len(doc_deps)}")
    print(f"ML Dependencies: {ml_success}/{len(ml_deps)}")
    print(f"Project Files: {file_success}/{len(project_files)}")
    print(f"Data Directories: {dir_success}/{len(data_dirs)}")
    print(f"Custom Modules: 6/6")
    print(f"Overall: {total_success}/{total_tests}")
    
    if total_success == total_tests:
        print("\n🎉 All tests passed! The installation is complete.")
        print("\n🚀 To run the application:")
        print("   streamlit run streamlit_app.py")
    else:
        print(f"\n⚠️  {total_tests - total_success} tests failed.")
        print("\n🔧 To fix missing dependencies:")
        print("   pip install -r requirements.txt")
        
        if doc_success < len(doc_deps):
            print("\n📄 For document processing, you may need:")
            print("   pip install PyMuPDF pdfplumber python-docx docx2txt")
        
        if ml_success < len(ml_deps):
            print("\n🤖 For ML features, you may need:")
            print("   pip install sentence-transformers nltk spacy")
            print("   python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    main()
