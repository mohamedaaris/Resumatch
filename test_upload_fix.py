#!/usr/bin/env python3
"""
Test script to verify the upload fix
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_openai_parser():
    """Test OpenAI parser with sample text"""
    print("Testing OpenAI Parser...")
    
    try:
        from openai_parser import OpenAIResumeParser
        
        # Sample resume text
        sample_text = """
        John Doe
        Email: john.doe@email.com
        Phone: +1-234-567-8900
        
        Skills: Python, Machine Learning, Data Analysis, SQL
        
        Experience:
        - Software Engineer at Tech Corp (2020-2023)
        - Data Analyst at Data Inc (2018-2020)
        
        Education:
        - Bachelor of Computer Science, University of Tech (2018)
        """
        
        parser = OpenAIResumeParser()
        print(f"   OpenAI available: {parser.use_openai}")
        
        if parser.use_openai:
            print("   Testing OpenAI parsing...")
            result = parser.parse_resume_with_openai(sample_text)
            print(f"   Parsed result keys: {list(result.keys())}")
            
            # Convert to legacy format
            legacy_result = parser.convert_to_legacy_format(result)
            print(f"   Legacy format keys: {list(legacy_result.keys())}")
            print("   [OK] OpenAI parsing successful")
        else:
            print("   [WARNING] OpenAI not available, testing fallback...")
            # Test fallback
            from enhanced_preprocess import EnhancedTextPreprocessor
            preprocessor = EnhancedTextPreprocessor()
            result = preprocessor.preprocess_resume(sample_text)
            print(f"   Fallback result keys: {list(result.keys())}")
            print("   [OK] Fallback parsing successful")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Parser test failed: {e}")
        return False

def test_app_import():
    """Test app import"""
    print("Testing app import...")
    
    try:
        from app import app, preprocessor
        print("   [OK] App imported successfully")
        print(f"   [OK] Preprocessor available: {preprocessor is not None}")
        return True
    except Exception as e:
        print(f"   [ERROR] App import failed: {e}")
        return False

def main():
    """Main test function"""
    print("ResuMatch AI - Upload Fix Test")
    print("=" * 40)
    
    # Test app import
    if not test_app_import():
        print("\n[ERROR] App import failed!")
        return False
    
    # Test OpenAI parser
    if not test_openai_parser():
        print("\n[ERROR] Parser test failed!")
        return False
    
    print("\n" + "=" * 40)
    print("[SUCCESS] All tests passed! Upload should work now.")
    print("   The application is running at: http://localhost:5000")
    print("   Try uploading a resume to test the fix.")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
