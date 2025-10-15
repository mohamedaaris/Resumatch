# ResuMatch-X: A Graph-Semantic and Explainable Internship Recommendation System with Adaptive Learning

ResuMatch-X upgrades the original ResuMatch AI into a research-grade framework that combines a hybrid Graph–Semantic matcher, domain-adaptive weighting, skill ontology normalization, explainable recommendations, and an interactive feedback loop.

## 🚀 Novelty & Features

- **Multi-format Resume Processing**: Supports PDF, DOCX, and image files (JPG, PNG, BMP, TIFF, GIF)
- **Advanced OCR**: Uses PyMuPDF and Tesseract for robust text extraction
- **NLP Processing**: spaCy-based text cleaning, tokenization, and Named Entity Recognition
- Enhanced recommendation algorithms using multiple approaches
- Semantic similarity matching with Sentence-BERT
- TF-IDF based keyword matching
- Combined scoring methods for optimal results

## 🛠️ Technology Stack

- Backend: Python, Flask
- OCR: PyMuPDF, pytesseract, Pillow
- NLP: spaCy (>=3.7), sentence-transformers (MiniLM)
- ML: scikit-learn, numpy; Graph: networkx
- Frontend: HTML/CSS/JS + Plotly for graph visualization
- Document Processing: python-docx

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
Resumatch_warp/
├── app.py
├── models/
│   ├── enhanced_model.py    # Enhanced matching algorithms
│   └── enhanced_preprocess.py # Enhanced text preprocessing
├── data/
│   └── sample_internships.json
├── templates/
│   └── ...
├── requirements.txt
└── ...
```

## 🔧 Usage

### Web Interface

1. **Home Page**: Overview of the system and features
2. **Upload Page**: Upload resume files for analysis
3. **Recommendations Page**: View matched internships with similarity scores

### API Endpoints

- `/upload` - Upload and process resume files
- `/recommendations` - View matched internships
- `/api/recommend` - API endpoint for recommendations
- `/api/jobs` - Get all available jobs
- `/health` - Health check endpoint

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

#### Enhanced Combined Method
- Combines TF-IDF and Sentence-BERT approaches
- Uses skill extraction and matching
- Provides detailed similarity scores and explanations

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

## 📈 Model Workflow Diagram

```mermaid
flowchart TD
  A[Resume Upload/OCR] --> B[NLP Preprocess (spaCy)]
  B --> C[Skills + Cleaned Text]
  C --> D[TF-IDF Vectorize]
  C --> E[MiniLM Embeddings]
  C --> F[Ontology Normalizer]
  D --> G[Graph Edges]
  E --> G
  F --> G
  G[Hybrid Graph + PageRank] --> H[Explainable Recommendations]
  H --> I[User Feedback]
  I --> J[Adaptive Weight Update (per-domain)]
  J --> G
```

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

