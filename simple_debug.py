#!/usr/bin/env python3
"""
Simple debug script for ResuMatch AI upload issues
"""

import os
import sys

def test_basic():
    """Test basic functionality"""
    print("ResuMatch AI - Simple Debug")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("ERROR: app.py not found. Make sure you're in the ResuMatch AI directory.")
        return False
    
    print("OK: Found app.py")
    
    # Check if templates exist
    if not os.path.exists('templates/upload.html'):
        print("ERROR: templates/upload.html not found.")
        return False
    
    print("OK: Found upload template")
    
    # Check if uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads', exist_ok=True)
        print("OK: Created uploads directory")
    else:
        print("OK: Uploads directory exists")
    
    # Test Flask app import
    try:
        from app import app, allowed_file, get_file_type
        print("OK: Flask app imported successfully")
    except Exception as e:
        print(f"ERROR importing Flask app: {e}")
        return False
    
    # Test file validation functions
    test_files = [
        ('test.pdf', True),
        ('test.docx', True),
        ('test.jpg', True),
        ('test.png', True),
        ('test.txt', False),
        ('test.exe', False)
    ]
    
    print("\nTesting file validation:")
    for filename, should_pass in test_files:
        result = allowed_file(filename)
        status = "OK" if result == should_pass else "ERROR"
        print(f"{status}: {filename} -> {result} (expected: {should_pass})")
    
    print("\n" + "=" * 40)
    print("Debug completed!")
    print("\nTo test upload:")
    print("1. Run: python run.py")
    print("2. Open: http://localhost:5000/upload")
    print("3. Try uploading a file")
    
    return True

if __name__ == '__main__':
    test_basic()
