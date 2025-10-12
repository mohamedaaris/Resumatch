#!/usr/bin/env python3
"""
Simple test for OpenAI integration
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_openai_direct():
    """Test OpenAI API directly"""
    print("Testing OpenAI API directly...")
    
    try:
        from openai import OpenAI
        from config import Config
        
        api_key = Config.get_openai_api_key()
        print(f"   API Key: {api_key[:20]}...")
        
        client = OpenAI(api_key=api_key)
        print("   [OK] OpenAI client created successfully")
        
        # Test a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello, respond with just 'Test successful'"}
            ],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"   [OK] API response: {result}")
        return True
        
    except Exception as e:
        print(f"   [ERROR] OpenAI test failed: {e}")
        return False

def test_parser_simple():
    """Test parser with simple text"""
    print("\nTesting parser with simple text...")
    
    try:
        from openai_parser import OpenAIResumeParser
        
        parser = OpenAIResumeParser()
        print(f"   Parser initialized: {parser.use_openai}")
        
        if parser.use_openai:
            # Simple test
            simple_text = "John Doe\nEmail: john@test.com\nSkills: Python, Java"
            result = parser.parse_resume_with_openai(simple_text)
            print(f"   [OK] Parsing successful, keys: {list(result.keys())}")
            return True
        else:
            print("   [WARNING] OpenAI not available")
            return False
            
    except Exception as e:
        print(f"   [ERROR] Parser test failed: {e}")
        return False

if __name__ == '__main__':
    print("Simple OpenAI Test")
    print("=" * 30)
    
    success1 = test_openai_direct()
    success2 = test_parser_simple()
    
    if success1 and success2:
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[ERROR] Some tests failed!")
