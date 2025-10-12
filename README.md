# ResuMatch AI

A smart internship recommendation system powered by OCR, NLP, and machine learning. ResuMatch AI analyzes uploaded resumes and matches them with internship descriptions using advanced text processing and similarity algorithms.

## 🚀 Features

- **Multi-format Resume Processing**: Supports PDF, DOCX, and image files (JPG, PNG, BMP, TIFF, GIF)
- **Advanced OCR**: Uses PyMuPDF and Tesseract for robust text extraction
- **NLP Processing**: spaCy-based text cleaning, tokenization, and Named Entity Recognition
- **Dual Similarity Matching**: 
  - TF-IDF vectorization for keyword-based matching
  - Sentence-BERT embeddings for semantic similarity
- **Web Interface**: Beautiful Bootstrap-based Flask application
- **Real-time Analysis**: Instant resume processing and recommendations
- **API Endpoints**: RESTful API for integration with other systems

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **OCR**: PyMuPDF, pytesseract, Pillow
- **NLP**: spaCy, sentence-transformers
- **ML**: scikit-learn, numpy, pandas
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Document Processing**: python-docx

## 📋 Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on your system
- spaCy English model

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd resumatch-ai
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install spaCy English model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Install Tesseract OCR**
   
   **Windows:**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Add to PATH or update the path in `extract_text.py`
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Linux:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```

## 🏃‍♂️ Quick Start

1. **Run the application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Upload a resume**
   - Go to the Upload page
   - Select a PDF, DOCX, or image file
   - Click "Analyze Resume"

4. **View recommendations**
   - See top 5 matching internships
   - Compare different analysis methods
   - View detailed job information

## 📁 Project Structure

```
resumatch-ai/
├── app.py                 # Main Flask application
├── extract_text.py        # OCR and document parsing
├── preprocess.py          # NLP text preprocessing
├── model.py              # Similarity matching models
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/
│   └── sample_internships.json  # Sample job data
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── upload.html
│   ├── recommendations.html
│   └── error.html
├── uploads/              # Uploaded files (created automatically)
├── logs/                 # Log files (created automatically)
└── results/              # Analysis results (created automatically)
```

## 🔧 Usage

### Web Interface

1. **Home Page**: Overview of the system and features
2. **Upload Page**: Upload resume files for analysis
3. **Recommendations Page**: View matched internships with similarity scores

### API Endpoints

- `GET /` - Home page
- `GET /upload` - Upload page
- `POST /upload` - Handle file upload
- `GET /recommendations` - View recommendations
- `POST /api/recommend` - API for recommendations
- `GET /api/jobs` - Get all available jobs
- `GET /health` - Health check

### API Usage Example

```python
import requests

# Upload resume text for analysis
response = requests.post('http://localhost:5000/api/recommend', 
                        json={
                            'resume_text': 'Your resume text here...',
                            'method': 'combined',  # or 'tfidf', 'sbert'
                            'top_k': 5
                        })

data = response.json()
recommendations = data['recommendations']
```

## 🧠 How It Works

### 1. Text Extraction
- **PDF**: Uses PyMuPDF for text extraction, falls back to OCR for scanned PDFs
- **DOCX**: Uses python-docx for document parsing
- **Images**: Uses Tesseract OCR for text recognition

### 2. NLP Preprocessing
- **Text Cleaning**: Remove special characters, normalize whitespace
- **Tokenization**: Split text into tokens using spaCy
- **Lemmatization**: Reduce words to their base forms
- **Named Entity Recognition**: Extract skills, education, experience
- **Skill Extraction**: Pattern-based skill detection

### 3. Similarity Matching

#### TF-IDF Method
- Creates TF-IDF vectors for resume and job descriptions
- Computes cosine similarity between vectors
- Good for keyword-based matching

#### Sentence-BERT Method
- Uses pre-trained sentence embeddings
- Captures semantic similarity
- Better for contextual understanding

#### Combined Method
- Weighted combination of both methods
- Default weights: 30% TF-IDF, 70% Sentence-BERT
- Provides balanced matching approach

## ⚙️ Configuration

Edit `config.py` to customize:

- File upload limits
- Model parameters
- Similarity weights
- Logging settings
- API configuration

## 📊 Sample Data

The system comes with sample internship data in `data/sample_internships.json`:

- Software Engineering Intern
- Data Science Intern
- Frontend Development Intern
- Backend Development Intern
- AI/ML Research Intern
- DevOps Intern
- Cybersecurity Intern
- Mobile Development Intern

## 🔍 Troubleshooting

### Common Issues

1. **Tesseract not found**
   - Install Tesseract OCR
   - Update path in `extract_text.py` if needed

2. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Memory issues with large files**
   - Reduce file size
   - Increase system memory
   - Use smaller models

4. **Low similarity scores**
   - Check if text was extracted correctly
   - Try different analysis methods
   - Ensure resume contains relevant keywords

### Logs

Check `logs/resumatch.log` for detailed error information.

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker (create Dockerfile)
docker build -t resumatch-ai .
docker run -p 5000:5000 resumatch-ai
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- spaCy for NLP processing
- Sentence-Transformers for semantic embeddings
- scikit-learn for machine learning utilities
- Flask for web framework
- Bootstrap for UI components

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

---

**ResuMatch AI** - Smart internship recommendations powered by AI 🚀

