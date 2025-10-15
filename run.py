#!/usr/bin/env python3
"""
ResuMatch AI - Main entry point
Run this script to start the ResuMatch AI application
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app import app, initialize_model
from config import config
from utils import setup_logging

def main():
    """Main function to run the ResuMatch AI application"""

    # Setup logging
    logger = setup_logging(
        log_level=config['default'].LOG_LEVEL,
        log_file=str(config['default'].LOG_FILE)
    )

    try:
        logger.info("Starting ResuMatch AI application...")

        # Create necessary directories
        os.makedirs('uploads', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)

        # Initialize model
        logger.info("Initializing ResuMatch model...")
        initialize_model()

        # Run the application
        logger.info("ResuMatch AI application started successfully!")
        logger.info("Access the application at: http://localhost:5000")

        app.run(
            debug=config['default'].DEBUG,
            host='0.0.0.0',
            port=5000
        )

    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        print(f"Error starting application: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Download spaCy model: python -m spacy download en_core_web_sm")
        print("3. Install Tesseract OCR on your system")
        sys.exit(1)

if __name__ == '__main__':
    main()
