"""
Similarity model for ResuMatch AI
Implements TF-IDF and Sentence-BERT embeddings for resume-job matching
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import logging
from typing import List, Dict, Tuple, Any
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResuMatchModel:
    """Main model class for resume-job matching using TF-IDF and Sentence-BERT"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the ResuMatch model
        
        Args:
            model_name (str): Name of the Sentence-BERT model to use
        """
        import os
        if model_name is None:
            try:
                from config import Config
                model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", getattr(Config, "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
            except Exception:
                model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model_name = model_name
        self.tfidf_vectorizer = None
        self.sentence_transformer = None
        self.job_descriptions = []
        self.job_embeddings_tfidf = None
        self.job_embeddings_sbert = None
        self.is_fitted = False
        
        logger.info(f"Initialized ResuMatch model with Sentence-BERT model: {model_name}")
    
def load_sentence_transformer(self):
        """Load the Sentence-BERT model (optionally using a Hugging Face token)"""
        try:
            logger.info(f"Loading Sentence-BERT model: {self.model_name}")
            import os
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if hf_token:
                try:
                    self.sentence_transformer = SentenceTransformer(self.model_name, use_auth_token=hf_token)
                except TypeError:
                    self.sentence_transformer = SentenceTransformer(self.model_name, token=hf_token)
            else:
                self.sentence_transformer = SentenceTransformer(self.model_name)
            logger.info("Sentence-BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Sentence-BERT model: {str(e)}")
            raise
    
    def load_job_descriptions(self, job_data: List[Dict[str, Any]]):
        """
        Load job descriptions from data
        
        Args:
            job_data (List[Dict[str, Any]]): List of job description dictionaries
        """
        try:
            logger.info(f"Loading {len(job_data)} job descriptions...")
            
            self.job_descriptions = []
            for job in job_data:
                # Combine title, description, and requirements into a single text
                combined_text = f"{job.get('title', '')} {job.get('description', '')} {' '.join(job.get('requirements', []))}"
                
                job_info = {
                    'id': job.get('id'),
                    'title': job.get('title', ''),
                    'company': job.get('company', ''),
                    'description': job.get('description', ''),
                    'requirements': job.get('requirements', []),
                    'location': job.get('location', ''),
                    'duration': job.get('duration', ''),
                    'combined_text': combined_text
                }
                self.job_descriptions.append(job_info)
            
            logger.info(f"Successfully loaded {len(self.job_descriptions)} job descriptions")
            
        except Exception as e:
            logger.error(f"Error loading job descriptions: {str(e)}")
            raise
    
    def fit_tfidf(self, texts: List[str]):
        """
        Fit TF-IDF vectorizer on job descriptions
        
        Args:
            texts (List[str]): List of job description texts
        """
        try:
            logger.info("Fitting TF-IDF vectorizer...")
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            self.job_embeddings_tfidf = self.tfidf_vectorizer.fit_transform(texts)
            logger.info(f"TF-IDF vectorizer fitted successfully. Shape: {self.job_embeddings_tfidf.shape}")
            
        except Exception as e:
            logger.error(f"Error fitting TF-IDF vectorizer: {str(e)}")
            raise
    
    def fit_sentence_transformer(self, texts: List[str]):
        """
        Generate Sentence-BERT embeddings for job descriptions
        
        Args:
            texts (List[str]): List of job description texts
        """
        try:
            logger.info("Generating Sentence-BERT embeddings...")
            
            if self.sentence_transformer is None:
                self.load_sentence_transformer()
            
            self.job_embeddings_sbert = self.sentence_transformer.encode(texts)
            logger.info(f"Sentence-BERT embeddings generated successfully. Shape: {self.job_embeddings_sbert.shape}")
            
        except Exception as e:
            logger.error(f"Error generating Sentence-BERT embeddings: {str(e)}")
            raise
    
    def fit(self, job_data: List[Dict[str, Any]]):
        """
        Fit the model on job descriptions
        
        Args:
            job_data (List[Dict[str, Any]]): List of job description dictionaries
        """
        try:
            logger.info("Fitting ResuMatch model...")
            
            # Load job descriptions
            self.load_job_descriptions(job_data)
            
            # Extract texts for vectorization
            texts = [job['combined_text'] for job in self.job_descriptions]
            
            # Fit TF-IDF
            self.fit_tfidf(texts)
            
            # Fit Sentence-BERT
            self.fit_sentence_transformer(texts)
            
            self.is_fitted = True
            logger.info("ResuMatch model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise
    
    def predict_tfidf(self, resume_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Predict job matches using TF-IDF similarity
        
        Args:
            resume_text (str): Resume text to match
            top_k (int): Number of top matches to return
            
        Returns:
            List[Dict[str, Any]]: List of top job matches with similarity scores
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            logger.info("Computing TF-IDF similarity...")
            
            # Transform resume text using fitted TF-IDF vectorizer
            resume_vector = self.tfidf_vectorizer.transform([resume_text])
            
            # Compute cosine similarity
            similarities = cosine_similarity(resume_vector, self.job_embeddings_tfidf).flatten()
            
            # Get top-k matches
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                job = self.job_descriptions[idx]
                result = {
                    'job_id': job['id'],
                    'title': job['title'],
                    'company': job['company'],
                    'description': job['description'],
                    'requirements': job['requirements'],
                    'location': job['location'],
                    'duration': job['duration'],
                    'similarity_score': float(similarities[idx]),
                    'similarity_percentage': float(similarities[idx] * 100),
                    'method': 'TF-IDF'
                }
                results.append(result)
            
            logger.info(f"TF-IDF prediction completed. Top similarity: {results[0]['similarity_percentage']:.2f}%")
            return results
            
        except Exception as e:
            logger.error(f"Error in TF-IDF prediction: {str(e)}")
            raise
    
    def predict_sentence_transformer(self, resume_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Predict job matches using Sentence-BERT similarity
        
        Args:
            resume_text (str): Resume text to match
            top_k (int): Number of top matches to return
            
        Returns:
            List[Dict[str, Any]]: List of top job matches with similarity scores
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            logger.info("Computing Sentence-BERT similarity...")
            
            # Generate embedding for resume text
            resume_embedding = self.sentence_transformer.encode([resume_text])
            
            # Compute cosine similarity
            similarities = cosine_similarity(resume_embedding, self.job_embeddings_sbert).flatten()
            
            # Get top-k matches
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                job = self.job_descriptions[idx]
                result = {
                    'job_id': job['id'],
                    'title': job['title'],
                    'company': job['company'],
                    'description': job['description'],
                    'requirements': job['requirements'],
                    'location': job['location'],
                    'duration': job['duration'],
                    'similarity_score': float(similarities[idx]),
                    'similarity_percentage': float(similarities[idx] * 100),
                    'method': 'Sentence-BERT'
                }
                results.append(result)
            
            logger.info(f"Sentence-BERT prediction completed. Top similarity: {results[0]['similarity_percentage']:.2f}%")
            return results
            
        except Exception as e:
            logger.error(f"Error in Sentence-BERT prediction: {str(e)}")
            raise
    
    def predict_combined(self, resume_text: str, top_k: int = 5, tfidf_weight: float = 0.3, sbert_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Predict job matches using combined TF-IDF and Sentence-BERT similarity
        
        Args:
            resume_text (str): Resume text to match
            top_k (int): Number of top matches to return
            tfidf_weight (float): Weight for TF-IDF similarity
            sbert_weight (float): Weight for Sentence-BERT similarity
            
        Returns:
            List[Dict[str, Any]]: List of top job matches with combined similarity scores
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            logger.info("Computing combined similarity...")
            
            # Get predictions from both methods
            tfidf_results = self.predict_tfidf(resume_text, top_k * 2)  # Get more for better combination
            sbert_results = self.predict_sentence_transformer(resume_text, top_k * 2)
            
            # Create a dictionary to store combined scores
            combined_scores = {}
            
            # Add TF-IDF scores
            for result in tfidf_results:
                job_id = result['job_id']
                combined_scores[job_id] = {
                    'job_info': result,
                    'tfidf_score': result['similarity_score'] * tfidf_weight,
                    'sbert_score': 0.0
                }
            
            # Add Sentence-BERT scores
            for result in sbert_results:
                job_id = result['job_id']
                if job_id in combined_scores:
                    combined_scores[job_id]['sbert_score'] = result['similarity_score'] * sbert_weight
                else:
                    combined_scores[job_id] = {
                        'job_info': result,
                        'tfidf_score': 0.0,
                        'sbert_score': result['similarity_score'] * sbert_weight
                    }
            
            # Calculate combined scores
            final_results = []
            for job_id, scores in combined_scores.items():
                combined_score = scores['tfidf_score'] + scores['sbert_score']
                job_info = scores['job_info'].copy()
                job_info['similarity_score'] = combined_score
                job_info['similarity_percentage'] = combined_score * 100
                job_info['method'] = 'Combined'
                job_info['tfidf_score'] = scores['tfidf_score']
                job_info['sbert_score'] = scores['sbert_score']
                final_results.append(job_info)
            
            # Sort by combined score and return top-k
            final_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            final_results = final_results[:top_k]
            
            logger.info(f"Combined prediction completed. Top similarity: {final_results[0]['similarity_percentage']:.2f}%")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in combined prediction: {str(e)}")
            raise
    
    def save_model(self, filepath: str):
        """
        Save the fitted model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        try:
            logger.info(f"Saving model to {filepath}...")
            
            model_data = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'job_descriptions': self.job_descriptions,
                'job_embeddings_tfidf': self.job_embeddings_tfidf,
                'job_embeddings_sbert': self.job_embeddings_sbert,
                'is_fitted': self.is_fitted,
                'model_name': self.model_name
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
        """
        Load a fitted model from disk
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            logger.info(f"Loading model from {filepath}...")
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.job_descriptions = model_data['job_descriptions']
            self.job_embeddings_tfidf = model_data['job_embeddings_tfidf']
            self.job_embeddings_sbert = model_data['job_embeddings_sbert']
            self.is_fitted = model_data['is_fitted']
            self.model_name = model_data['model_name']
            
            # Load Sentence-BERT model
            self.load_sentence_transformer()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


def load_job_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load job data from JSON file
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        List[Dict[str, Any]]: List of job descriptions
    """
    try:
        logger.info(f"Loading job data from {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            job_data = json.load(f)
        
        logger.info(f"Successfully loaded {len(job_data)} job descriptions")
        return job_data
        
    except Exception as e:
        logger.error(f"Error loading job data: {str(e)}")
        raise


def create_and_fit_model(job_data_file: str, model_save_path: str = None) -> ResuMatchModel:
    """
    Create and fit a ResuMatch model from job data
    
    Args:
        job_data_file (str): Path to the job data JSON file
        model_save_path (str, optional): Path to save the fitted model
        
    Returns:
        ResuMatchModel: Fitted model
    """
    try:
        logger.info("Creating and fitting ResuMatch model...")
        
        # Load job data
        job_data = load_job_data(job_data_file)
        
        # Create and fit model
        model = ResuMatchModel()
        model.fit(job_data)
        
        # Save model if path provided
        if model_save_path:
            model.save_model(model_save_path)
        
        logger.info("Model creation and fitting completed successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error creating and fitting model: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the ResuMatch model
    print("ResuMatch AI - Similarity Model")
    print("Testing model creation and fitting...")
    
    try:
        # Create and fit model
        model = create_and_fit_model('data/sample_internships.json', 'model.pkl')
        
        # Test prediction
        sample_resume = """
        John Smith
        Software Engineer
        Experience: 3 years in Python, JavaScript, and web development
        Education: Bachelor's in Computer Science from MIT
        Skills: Machine Learning, Data Science, React, Node.js, SQL, AWS
        """
        
        print("\nTesting predictions...")
        print("TF-IDF Results:")
        tfidf_results = model.predict_tfidf(sample_resume, top_k=3)
        for result in tfidf_results:
            print(f"- {result['title']} at {result['company']}: {result['similarity_percentage']:.2f}%")
        
        print("\nSentence-BERT Results:")
        sbert_results = model.predict_sentence_transformer(sample_resume, top_k=3)
        for result in sbert_results:
            print(f"- {result['title']} at {result['company']}: {result['similarity_percentage']:.2f}%")
        
        print("\nCombined Results:")
        combined_results = model.predict_combined(sample_resume, top_k=3)
        for result in combined_results:
            print(f"- {result['title']} at {result['company']}: {result['similarity_percentage']:.2f}%")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to install required dependencies and download spaCy model:")
        print("pip install -r requirements.txt")
        print("python -m spacy download en_core_web_sm")
