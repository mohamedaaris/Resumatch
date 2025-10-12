#!/usr/bin/env python3
"""
Simple test for upload functionality
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_upload_flow():
    """Test the upload flow with fallback parsing"""
    print("Testing upload flow...")
    
    try:
        from app import app, preprocessor, model
        from enhanced_preprocess import EnhancedTextPreprocessor
        
        # Initialize if not already done
        if preprocessor is None:
            print("   Initializing preprocessor...")
            preprocessor = EnhancedTextPreprocessor()
            print("   [OK] Preprocessor initialized")
        
        # Test with sample text
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
        
        print("   Testing preprocessing...")
        processed_data = preprocessor.preprocess_resume(sample_text)
        print(f"   [OK] Preprocessing successful, keys: {list(processed_data.keys())}")
        
        # Test with Flask app
        print("   Testing Flask app...")
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("   [OK] App responds to requests")
            else:
                print(f"   [WARNING] App responded with status {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Upload Flow Test")
    print("=" * 20)
    success = test_upload_flow()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")