"""
Enhanced similarity model for ResuMatch AI
Implements advanced matching with location preferences, skill gap analysis, and learning suggestions
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
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedResuMatchModel:
    """Enhanced model class for resume-job matching with advanced features"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the Enhanced ResuMatch model
        
        Args:
            model_name (str): Name of the Sentence-BERT model to use
        """
        import os
        # Prefer explicit arg, then env/config, then sensible default
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
        
        # Indian cities for location matching
        self.indian_cities = {
            'bangalore', 'bengaluru', 'mumbai', 'delhi', 'hyderabad', 'chennai', 'pune', 'kolkata', 'ahmedabad',
            'gurgaon', 'gurugram', 'noida', 'faridabad', 'ghaziabad', 'jaipur', 'lucknow', 'kanpur', 'nagpur',
            'indore', 'bhopal', 'visakhapatnam', 'vijayawada', 'coimbatore', 'madurai', 'salem', 'tiruchirappalli',
            'kochi', 'thiruvananthapuram', 'calicut', 'kozhikode', 'mysore', 'mysuru', 'mangalore', 'hubli'
        }
        
        logger.info(f"Initialized Enhanced ResuMatch model with Sentence-BERT model: {model_name}")
    
    def load_sentence_transformer(self):
        """Load the Sentence-BERT model (optionally using a Hugging Face token)"""
        try:
            logger.info(f"Loading Sentence-BERT model: {self.model_name}")
            import os
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if hf_token:
                try:
                    # Older API
                    self.sentence_transformer = SentenceTransformer(self.model_name, use_auth_token=hf_token)
                except TypeError:
                    # Newer API may use `token` kwarg
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
                    'skills_needed': job.get('skills_needed', []),
                    'location': job.get('location', ''),
                    'duration': job.get('duration', ''),
                    'field': job.get('field', ''),
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
            logger.info("Fitting Enhanced ResuMatch model...")
            
            # Load job descriptions
            self.load_job_descriptions(job_data)
            
            # Extract texts for vectorization
            texts = [job['combined_text'] for job in self.job_descriptions]
            
            # Fit TF-IDF
            self.fit_tfidf(texts)
            
            # Fit Sentence-BERT
            self.fit_sentence_transformer(texts)
            
            self.is_fitted = True
            logger.info("Enhanced ResuMatch model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise
    
    def calculate_location_score(self, resume_locations: List[str], job_location: str) -> float:
        """
        Calculate location compatibility score
        
        Args:
            resume_locations (List[str]): Locations mentioned in resume
            job_location (str): Job location
            
        Returns:
            float: Location compatibility score (0-1)
        """
        try:
            if not resume_locations or not job_location:
                return 0.5  # Neutral score if no location info
            
            job_location_lower = job_location.lower()
            resume_locations_lower = [loc.lower() for loc in resume_locations]
            
            # Check for exact match
            for resume_loc in resume_locations_lower:
                if resume_loc in job_location_lower or job_location_lower in resume_loc:
                    return 1.0
            
            # Check for Indian city match
            job_city = None
            for city in self.indian_cities:
                if city in job_location_lower:
                    job_city = city
                    break
            
            if job_city:
                for resume_loc in resume_locations_lower:
                    if city in resume_loc:
                        return 0.8  # High score for same city
            
            # Check for country match
            if 'india' in job_location_lower:
                for resume_loc in resume_locations_lower:
                    if 'india' in resume_loc:
                        return 0.6  # Medium score for same country
            
            return 0.3  # Low score for different locations
            
        except Exception as e:
            logger.error(f"Error calculating location score: {str(e)}")
            return 0.5
    
    def calculate_skill_match_score(self, resume_skills: List[str], job_skills: List[str]) -> Tuple[float, List[str], List[str]]:
        """
        Calculate skill match score and identify missing skills
        
        Args:
            resume_skills (List[str]): Skills from resume
            job_skills (List[str]): Skills required for job
            
        Returns:
            Tuple[float, List[str], List[str]]: (match_score, matched_skills, missing_skills)
        """
        try:
            if not job_skills:
                return 1.0, resume_skills, []
            
            resume_skills_lower = [skill.lower() for skill in resume_skills]
            job_skills_lower = [skill.lower() for skill in job_skills]
            
            matched_skills = []
            missing_skills = []
            
            # Find matched skills
            for job_skill in job_skills:
                job_skill_lower = job_skill.lower()
                for resume_skill in resume_skills:
                    if (job_skill_lower in resume_skill.lower() or 
                        resume_skill.lower() in job_skill_lower or
                        self.are_skills_similar(job_skill_lower, resume_skill.lower())):
                        matched_skills.append(job_skill)
                        break
                else:
                    missing_skills.append(job_skill)
            
            # Calculate match score
            match_score = len(matched_skills) / len(job_skills) if job_skills else 1.0
            
            return match_score, matched_skills, missing_skills
            
        except Exception as e:
            logger.error(f"Error calculating skill match score: {str(e)}")
            return 0.0, [], []
    
    def are_skills_similar(self, skill1: str, skill2: str) -> bool:
        """
        Check if two skills are similar (for fuzzy matching)
        
        Args:
            skill1 (str): First skill
            skill2 (str): Second skill
            
        Returns:
            bool: True if skills are similar
        """
        # Skill similarity mappings
        skill_synonyms = {
            'javascript': ['js', 'ecmascript'],
            'python': ['py'],
            'machine learning': ['ml', 'artificial intelligence', 'ai'],
            'data science': ['data analysis', 'analytics'],
            'web development': ['web dev', 'frontend', 'backend'],
            'mobile development': ['mobile dev', 'android', 'ios'],
            'cloud computing': ['cloud', 'aws', 'azure', 'gcp'],
            'database': ['db', 'sql', 'mysql', 'postgresql'],
            'version control': ['git', 'github', 'gitlab'],
            'testing': ['qa', 'quality assurance', 'test automation']
        }
        
        for main_skill, synonyms in skill_synonyms.items():
            if skill1 in synonyms and skill2 in synonyms:
                return True
            if skill1 == main_skill and skill2 in synonyms:
                return True
            if skill2 == main_skill and skill1 in synonyms:
                return True
        
        return False
    
    def generate_learning_suggestions(self, missing_skills: List[str], job_field: str) -> List[Dict[str, str]]:
        """
        Generate learning suggestions for missing skills
        
        Args:
            missing_skills (List[str]): Skills that need to be learned
            job_field (str): Field of the job
            
        Returns:
            List[Dict[str, str]]: Learning suggestions with resources
        """
        try:
            suggestions = []
            
            # Learning resources mapping
            learning_resources = {
                'python': {
                    'course': 'Python for Data Science - Coursera',
                    'tutorial': 'Python Tutorial - W3Schools',
                    'practice': 'LeetCode Python Problems',
                    'project': 'Build a web scraper or data analysis project'
                },
                'javascript': {
                    'course': 'JavaScript Complete Course - Udemy',
                    'tutorial': 'MDN JavaScript Guide',
                    'practice': 'Build interactive web applications',
                    'project': 'Create a portfolio website'
                },
                'machine learning': {
                    'course': 'Machine Learning Course - Andrew Ng (Coursera)',
                    'tutorial': 'Scikit-learn Documentation',
                    'practice': 'Kaggle competitions',
                    'project': 'Build a prediction model'
                },
                'aws': {
                    'course': 'AWS Certified Solutions Architect - A Cloud Guru',
                    'tutorial': 'AWS Documentation',
                    'practice': 'AWS Free Tier projects',
                    'project': 'Deploy a web application on AWS'
                },
                'react': {
                    'course': 'React Complete Course - Udemy',
                    'tutorial': 'React Official Documentation',
                    'practice': 'Build React components',
                    'project': 'Create a React-based web application'
                },
                'sql': {
                    'course': 'SQL for Data Analysis - Coursera',
                    'tutorial': 'SQL Tutorial - W3Schools',
                    'practice': 'HackerRank SQL problems',
                    'project': 'Design and query a database'
                },
                'docker': {
                    'course': 'Docker Complete Course - Udemy',
                    'tutorial': 'Docker Official Documentation',
                    'practice': 'Containerize applications',
                    'project': 'Dockerize a web application'
                },
                'kubernetes': {
                    'course': 'Kubernetes Complete Course - Udemy',
                    'tutorial': 'Kubernetes Documentation',
                    'practice': 'Minikube local development',
                    'project': 'Deploy microservices with Kubernetes'
                }
            }
            
            for skill in missing_skills[:5]:  # Limit to top 5 missing skills
                skill_lower = skill.lower()
                if skill_lower in learning_resources:
                    suggestions.append({
                        'skill': skill,
                        'resources': learning_resources[skill_lower],
                        'estimated_time': self.get_learning_time(skill_lower),
                        'difficulty': self.get_skill_difficulty(skill_lower)
                    })
                else:
                    # Generic suggestion for unknown skills
                    suggestions.append({
                        'skill': skill,
                        'resources': {
                            'course': f'{skill} Complete Course - Udemy',
                            'tutorial': f'{skill} Documentation',
                            'practice': f'Practice {skill} projects',
                            'project': f'Build a {skill} project'
                        },
                        'estimated_time': '2-4 weeks',
                        'difficulty': 'Intermediate'
                    })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating learning suggestions: {str(e)}")
            return []
    
    def get_learning_time(self, skill: str) -> str:
        """Get estimated learning time for a skill"""
        time_mapping = {
            'python': '3-4 weeks',
            'javascript': '4-6 weeks',
            'machine learning': '8-12 weeks',
            'aws': '6-8 weeks',
            'react': '4-6 weeks',
            'sql': '2-3 weeks',
            'docker': '2-3 weeks',
            'kubernetes': '4-6 weeks'
        }
        return time_mapping.get(skill, '3-4 weeks')
    
    def get_skill_difficulty(self, skill: str) -> str:
        """Get difficulty level for a skill"""
        difficulty_mapping = {
            'python': 'Beginner',
            'javascript': 'Intermediate',
            'machine learning': 'Advanced',
            'aws': 'Intermediate',
            'react': 'Intermediate',
            'sql': 'Beginner',
            'docker': 'Intermediate',
            'kubernetes': 'Advanced'
        }
        return difficulty_mapping.get(skill, 'Intermediate')
    
    def predict_enhanced(self, resume_data: Dict[str, any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Enhanced prediction with location, skill gap analysis, and learning suggestions
        
        Args:
            resume_data (Dict[str, any]): Preprocessed resume data
            top_k (int): Number of top matches to return
            
        Returns:
            List[Dict[str, Any]]: List of enhanced job matches
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            logger.info("Computing enhanced similarity...")
            
            resume_text = resume_data['cleaned_text']
            resume_skills = resume_data.get('skills', [])
            resume_locations = resume_data.get('locations', [])
            resume_field = resume_data.get('field_of_interest', [])
            
            # Get basic similarity scores
            tfidf_results = self.predict_tfidf(resume_text, top_k * 2)
            sbert_results = self.predict_sentence_transformer(resume_text, top_k * 2)
            
            # Create enhanced results
            enhanced_results = []
            
            for i, job in enumerate(self.job_descriptions):
                # Get similarity scores
                tfidf_score = 0.0
                sbert_score = 0.0
                
                for result in tfidf_results:
                    if result['job_id'] == job['id']:
                        tfidf_score = result['similarity_score']
                        break
                
                for result in sbert_results:
                    if result['job_id'] == job['id']:
                        sbert_score = result['similarity_score']
                        break
                
                # Calculate location score
                location_score = self.calculate_location_score(resume_locations, job['location'])
                
                # Calculate skill match
                skill_match_score, matched_skills, missing_skills = self.calculate_skill_match_score(
                    resume_skills, job.get('skills_needed', [])
                )
                
                # Calculate field match
                field_match_score = 0.5  # Default
                if resume_field and job.get('field'):
                    if any(field.lower() in job['field'].lower() for field in resume_field):
                        field_match_score = 1.0
                
                # Calculate combined score with weights
                combined_score = (
                    tfidf_score * 0.25 +
                    sbert_score * 0.35 +
                    location_score * 0.15 +
                    skill_match_score * 0.20 +
                    field_match_score * 0.05
                )
                
                # Generate learning suggestions for low-scoring jobs
                learning_suggestions = []
                if combined_score < 0.6 and missing_skills:
                    learning_suggestions = self.generate_learning_suggestions(
                        missing_skills, job.get('field', '')
                    )
                
                enhanced_result = {
                    'job_id': job['id'],
                    'title': job['title'],
                    'company': job['company'],
                    'description': job['description'],
                    'requirements': job['requirements'],
                    'skills_needed': job.get('skills_needed', []),
                    'location': job['location'],
                    'duration': job['duration'],
                    'field': job.get('field', ''),
                    'similarity_score': combined_score,
                    'similarity_percentage': combined_score * 100,
                    'method': 'Enhanced',
                    'breakdown': {
                        'tfidf_score': tfidf_score,
                        'sbert_score': sbert_score,
                        'location_score': location_score,
                        'skill_match_score': skill_match_score,
                        'field_match_score': field_match_score
                    },
                    'skill_analysis': {
                        'matched_skills': matched_skills,
                        'missing_skills': missing_skills,
                        'skill_match_percentage': skill_match_score * 100
                    },
                    'learning_suggestions': learning_suggestions,
                    'location_preference': location_score > 0.7
                }
                
                enhanced_results.append(enhanced_result)
            
            # Sort by combined score and return top-k
            enhanced_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            enhanced_results = enhanced_results[:top_k]
            
            logger.info(f"Enhanced prediction completed. Top similarity: {enhanced_results[0]['similarity_percentage']:.2f}%")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction: {str(e)}")
            raise
    
    def predict_tfidf(self, resume_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Predict job matches using TF-IDF similarity"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
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
            
            return results
            
        except Exception as e:
            logger.error(f"Error in TF-IDF prediction: {str(e)}")
            raise
    
    def predict_sentence_transformer(self, resume_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Predict job matches using Sentence-BERT similarity"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
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
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Sentence-BERT prediction: {str(e)}")
            raise


# Backward compatibility
class ResuMatchModel(EnhancedResuMatchModel):
    """Backward compatibility wrapper"""
    pass


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


def create_and_fit_model(job_data_file: str, model_save_path: str = None) -> EnhancedResuMatchModel:
    """
    Create and fit an Enhanced ResuMatch model from job data
    
    Args:
        job_data_file (str): Path to the job data JSON file
        model_save_path (str, optional): Path to save the fitted model
        
    Returns:
        EnhancedResuMatchModel: Fitted model
    """
    try:
        logger.info("Creating and fitting Enhanced ResuMatch model...")
        
        # Load job data
        job_data = load_job_data(job_data_file)
        
        # Create and fit model
        model = EnhancedResuMatchModel()
        model.fit(job_data)
        
        # Save model if path provided
        if model_save_path:
            model.save_model(model_save_path)
        
        logger.info("Enhanced model creation and fitting completed successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error creating and fitting model: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the Enhanced ResuMatch model
    print("ResuMatch AI - Enhanced Similarity Model")
    print("Testing enhanced model creation and fitting...")
    
    try:
        # Create and fit model
        model = create_and_fit_model('data/sample_internships.json', 'enhanced_model.pkl')
        
        # Test enhanced prediction
        sample_resume_data = {
            'cleaned_text': """
            John Smith
            Software Engineer
            Experience: 3 years in Python, JavaScript, and web development
            Education: Bachelor's in Computer Science from MIT
            Skills: Machine Learning, Data Science, React, Node.js, SQL, AWS
            Location: Bangalore, India
            Certifications: AWS Certified Solutions Architect
            Projects: E-commerce website using React and Node.js
            Achievements: Won first place in hackathon
            Field of Interest: Software Engineering, Data Science
            """,
            'skills': ['python', 'javascript', 'react', 'node.js', 'sql', 'aws', 'machine learning', 'data science'],
            'locations': ['bangalore', 'india'],
            'certifications': ['aws certified solutions architect'],
            'projects': ['e-commerce website'],
            'achievements': ['won first place in hackathon'],
            'field_of_interest': ['software engineering', 'data science']
        }
        
        print("\nTesting enhanced predictions...")
        enhanced_results = model.predict_enhanced(sample_resume_data, top_k=3)
        
        for result in enhanced_results:
            print(f"\n- {result['title']} at {result['company']}: {result['similarity_percentage']:.2f}%")
            print(f"  Location Score: {result['breakdown']['location_score']:.2f}")
            print(f"  Skill Match: {result['skill_analysis']['skill_match_percentage']:.2f}%")
            print(f"  Missing Skills: {result['skill_analysis']['missing_skills']}")
            if result['learning_suggestions']:
                print(f"  Learning Suggestions: {len(result['learning_suggestions'])} skills to learn")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to install required dependencies and download spaCy model:")
        print("pip install -r requirements.txt")
        print("python -m spacy download en_core_web_sm")
