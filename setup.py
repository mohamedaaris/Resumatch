#!/usr/bin/env python3
"""
Setup script for ResuMatch AI
Automates the setup process for the ResuMatch AI application
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    return run_command("pip install -r requirements.txt", "Installing Python packages")

def download_spacy_model():
    """Download spaCy English model"""
    print("\nDownloading spaCy English model...")
    return run_command("python -m spacy download en_core_web_sm", "Downloading spaCy model")

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    directories = ['uploads', 'logs', 'results', 'data']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def check_tesseract():
    """Check if Tesseract is installed"""
    print("\nChecking Tesseract OCR...")
    try:
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Tesseract OCR is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("✗ Tesseract OCR not found")
    print("\nPlease install Tesseract OCR:")
    
    system = platform.system().lower()
    if system == "windows":
        print("Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("Or use: choco install tesseract")
    elif system == "darwin":  # macOS
        print("macOS: brew install tesseract")
    elif system == "linux":
        print("Linux: sudo apt-get install tesseract-ocr")
    
    return False

def run_tests():
    """Run system tests"""
    print("\nRunning system tests...")
    return run_command("python test_app.py", "Running system tests")

def main():
    """Main setup function"""
    print("ResuMatch AI - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\nFailed to install dependencies. Please check the error messages above.")
        sys.exit(1)
    
    # Download spaCy model
    if not download_spacy_model():
        print("\nFailed to download spaCy model. Please run manually:")
        print("python -m spacy download en_core_web_sm")
    
    # Check Tesseract
    tesseract_ok = check_tesseract()
    
    # Run tests
    print("\n" + "=" * 50)
    print("Setup completed! Running tests...")
    
    if run_tests():
        print("\n" + "=" * 50)
        print("🎉 ResuMatch AI setup completed successfully!")
        print("\nTo start the application, run:")
        print("python run.py")
        print("\nOr for development:")
        print("python app.py")
        
        if not tesseract_ok:
            print("\n⚠️  Note: Tesseract OCR is not installed.")
            print("Some features may not work properly.")
            print("Please install Tesseract OCR for full functionality.")
    else:
        print("\n" + "=" * 50)
        print("❌ Setup completed but some tests failed.")
        print("Please check the error messages above and fix any issues.")
        print("You can run tests manually with: python test_app.py")

if __name__ == '__main__':
    main()
