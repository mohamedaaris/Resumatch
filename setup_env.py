#!/usr/bin/env python3
"""
Environment setup script for ResuMatch AI
"""

import os
import sys

def setup_environment():
    """Setup environment for ResuMatch AI"""
    print("🚀 ResuMatch AI - Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print("✅ Python version:", sys.version.split()[0])
    
    # Check for OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("\n⚠️  OpenAI API Key not found!")
        print("   To enable advanced resume parsing with AI:")
        print("   1. Get your API key from: https://platform.openai.com/api-keys")
        print("   2. Set it as an environment variable:")
        print("      Windows: set OPENAI_API_KEY=your-api-key-here")
        print("      Linux/Mac: export OPENAI_API_KEY=your-api-key-here")
        print("   3. Or create a .env file with: OPENAI_API_KEY=your-api-key-here")
        print("\n   The system will work without it, but with basic parsing only.")
    else:
        print("✅ OpenAI API key found - Advanced AI parsing enabled!")
    
    # Check required directories
    required_dirs = ['uploads', 'data', 'templates']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"✅ Created directory: {dir_name}")
        else:
            print(f"✅ Directory exists: {dir_name}")
    
    # Check required files
    required_files = [
        'data/sample_internships.json',
        'templates/base.html',
        'templates/index.html',
        'templates/upload.html',
        'templates/profile.html',
        'templates/recommendations.html'
    ]
    
    missing_files = []
    for file_name in required_files:
        if not os.path.exists(file_name):
            missing_files.append(file_name)
        else:
            print(f"✅ File exists: {file_name}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        print("   Please ensure all required files are present.")
        return False
    
    print("\n🎉 Environment setup complete!")
    print("   Run 'python run.py' to start the application.")
    return True

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)
