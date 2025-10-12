# ResuMatch AI - Installation Guide

## Quick Start

1. **Clone or download the project**
   ```bash
   # If you have git
   git clone <repository-url>
   cd resumatch-ai
   
   # Or download and extract the ZIP file
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```
   
   This will automatically:
   - Install all Python dependencies
   - Download the spaCy English model
   - Create necessary directories
   - Run system tests

3. **Start the application**
   ```bash
   python run.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

## Manual Installation

If the automated setup doesn't work, follow these steps:

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 3. Install Tesseract OCR

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install and add to PATH
- Or use: `choco install tesseract`

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

### 4. Create Directories
```bash
mkdir uploads logs results
```

### 5. Test Installation
```bash
python test_app.py
```

### 6. Run Application
```bash
python run.py
```

## Troubleshooting

### Common Issues

1. **"spaCy model not found"**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **"Tesseract not found"**
   - Install Tesseract OCR (see above)
   - Update path in `extract_text.py` if needed

3. **"Permission denied"**
   - Run with administrator/sudo privileges
   - Check file permissions

4. **"Module not found"**
   - Make sure you're in the project directory
   - Check if all dependencies are installed

5. **"Port already in use"**
   - Change port in `run.py` or `app.py`
   - Kill existing processes using port 5000

### System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection for downloading models

### Supported File Formats

- **PDF**: Native text extraction + OCR fallback
- **DOCX**: Microsoft Word documents
- **Images**: JPG, PNG, BMP, TIFF, GIF

### Performance Tips

- Use smaller files for faster processing
- Close other applications to free up memory
- Use SSD storage for better performance
- Ensure stable internet connection for model downloads

## Development Setup

For development and customization:

1. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8
   ```

2. **Run tests**
   ```bash
   python test_app.py
   ```

3. **Code formatting**
   ```bash
   black *.py
   flake8 *.py
   ```

4. **Run in debug mode**
   ```bash
   python app.py
   ```

## Production Deployment

For production deployment:

1. **Use a production WSGI server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Set environment variables**
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secret-key
   ```

3. **Use a reverse proxy** (nginx, Apache)
4. **Set up SSL certificates**
5. **Configure logging and monitoring**

## Support

If you encounter issues:

1. Check the logs in `logs/resumatch.log`
2. Run `python test_app.py` to diagnose problems
3. Check system requirements
4. Ensure all dependencies are installed
5. Verify Tesseract OCR is working

For additional help, check the README.md file or create an issue on GitHub.
