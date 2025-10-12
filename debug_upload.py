#!/usr/bin/env python3
"""
Debug script for ResuMatch AI upload issues
This script helps diagnose file upload problems
"""

import os
import sys
from pathlib import Path

def test_file_upload():
    """Test file upload functionality"""
    print("ResuMatch AI - Upload Debug Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found. Make sure you're in the ResuMatch AI directory.")
        return False
    
    print("✓ Found app.py in current directory")
    
    # Check if templates exist
    if not os.path.exists('templates/upload.html'):
        print("❌ Error: templates/upload.html not found.")
        return False
    
    print("✓ Found upload template")
    
    # Check if uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads', exist_ok=True)
        print("✓ Created uploads directory")
    else:
        print("✓ Uploads directory exists")
    
    # Test Flask app import
    try:
        from app import app, allowed_file, get_file_type
        print("✓ Flask app imported successfully")
    except Exception as e:
        print(f"❌ Error importing Flask app: {e}")
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
        status = "✓" if result == should_pass else "❌"
        print(f"{status} {filename}: {result} (expected: {should_pass})")
    
    # Test file type extraction
    print("\nTesting file type extraction:")
    for filename, _ in test_files:
        if '.' in filename:
            file_type = get_file_type(filename)
            print(f"✓ {filename} -> {file_type}")
    
    print("\n" + "=" * 50)
    print("Upload debug completed!")
    print("\nTo test the upload functionality:")
    print("1. Run: python run.py")
    print("2. Open: http://localhost:5000/upload")
    print("3. Try uploading a PDF, DOCX, or image file")
    print("4. Check the browser console for JavaScript errors")
    print("5. Check the terminal for Python logs")
    
    return True

if __name__ == '__main__':
    test_file_upload()
