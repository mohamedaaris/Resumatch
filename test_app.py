#!/usr/bin/env python3
"""
Test script for ResuMatch AI
Tests the application startup and basic functionality
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from config import config, Config
        print("[OK] Config imported successfully")
        if Config.get_openai_api_key():
            print("   OpenAI API Key detected (env)")
        else:
            print("   OpenAI API Key not set")
        
        from app import app
        print("[OK] App imported successfully")
        
        from openai_parser import OpenAIResumeParser
        print("[OK] OpenAI Parser imported successfully")
        
        from enhanced_preprocess import EnhancedTextPreprocessor
        print("[OK] Enhanced Preprocessor imported successfully")
        
        from enhanced_model import EnhancedResuMatchModel
        print("[OK] Enhanced Model imported successfully")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Import error: {e}")
        return False

def test_openai_parser():
    """Test OpenAI parser initialization"""
    print("\nTesting OpenAI Parser...")
    
    try:
        from openai_parser import OpenAIResumeParser
        parser = OpenAIResumeParser()
        
        if parser.use_openai:
            print("[OK] OpenAI Parser initialized with API key")
        else:
            print("[WARNING] OpenAI Parser initialized without API key")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] OpenAI Parser error: {e}")
        return False

def test_app_startup():
    """Test Flask app startup"""
    print("\nTesting Flask app startup...")
    
    try:
        from app import app, initialize_model
        
        # Test model initialization
        print("   Initializing model...")
        initialize_model()
        print("[OK] Model initialized successfully")
        
        # Test app configuration
        print("   Testing app configuration...")
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("[OK] App responds to requests")
            else:
                print(f"[WARNING] App responded with status {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] App startup error: {e}")
        return False

def main():
    """Main test function"""
    print("ResuMatch AI - Test Suite")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n[ERROR] Import tests failed!")
        return False
    
    # Test OpenAI parser
    if not test_openai_parser():
        print("\n[ERROR] OpenAI Parser tests failed!")
        return False
    
    # Test app startup
    if not test_app_startup():
        print("\n[ERROR] App startup tests failed!")
        return False
    
    print("\n" + "=" * 40)
    print("[SUCCESS] All tests passed! ResuMatch AI is ready to run.")
    print("   Run 'python run.py' to start the application.")
    print("   Access it at: http://localhost:5000")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)