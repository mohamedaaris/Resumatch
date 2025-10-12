"""
NLP preprocessing module for ResuMatch AI
Handles text cleaning, tokenization, lemmatization, and Named Entity Recognition
"""

import spacy
import re
import logging
from typing import List, Dict, Set, Tuple
import string
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Handles NLP preprocessing of resume and job description text"""
    
    def __init__(self):
        """Initialize the text preprocessor with spaCy model"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        try:
            logger.info("Cleaning text...")
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep alphanumeric, spaces, and common punctuation
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', ' ', text)
            
            # Remove multiple consecutive punctuation
            text = re.sub(r'[\.\,\;\:\!\?]{2,}', '.', text)
            
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text)
            
            # Strip leading/trailing whitespace
            text = text.strip()
            
            logger.info(f"Text cleaned successfully ({len(text)} characters)")
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            raise
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy NER
        
        Args:
            text (str): Text to extract entities from
            
        Returns:
            Dict[str, List[str]]: Dictionary with entity types as keys and lists of entities as values
        """
        try:
            logger.info("Extracting named entities...")
            
            doc = self.nlp(text)
            entities = {
                'PERSON': [],
                'ORG': [],
                'GPE': [],  # Geopolitical entities (countries, cities, states)
                'DATE': [],
                'MONEY': [],
                'PERCENT': [],
                'TIME': [],
                'EVENT': [],
                'FAC': [],  # Facilities
                'LANGUAGE': [],
                'LAW': [],
                'LOC': [],  # Locations
                'NORP': [],  # Nationalities, religious or political groups
                'PRODUCT': [],
                'WORK_OF_ART': []
            }
            
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text.strip())
            
            # Remove duplicates while preserving order
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))
            
            logger.info(f"Extracted entities: {sum(len(v) for v in entities.values())} total")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            raise
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract potential skills from text using pattern matching and NER
        
        Args:
            text (str): Text to extract skills from
            
        Returns:
            List[str]: List of potential skills
        """
        try:
            logger.info("Extracting skills...")
            
            # Common technical skills patterns
            skill_patterns = [
                r'\b(python|java|javascript|typescript|react|angular|vue|node\.?js|express)\b',
                r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch)\b',
                r'\b(aws|azure|gcp|docker|kubernetes|jenkins|git|github|gitlab)\b',
                r'\b(machine learning|ml|deep learning|ai|artificial intelligence|nlp|natural language processing)\b',
                r'\b(data science|data analysis|statistics|pandas|numpy|scikit-learn|tensorflow|pytorch)\b',
                r'\b(html|css|bootstrap|sass|less|webpack|babel)\b',
                r'\b(rest api|graphql|microservices|agile|scrum|devops)\b',
                r'\b(linux|unix|bash|powershell|automation|testing|unit testing)\b'
            ]
            
            skills = set()
            
            # Extract using patterns
            for pattern in skill_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                skills.update(matches)
            
            # Extract using spaCy NER for technical terms
            doc = self.nlp(text)
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
                    # Check if it's likely a technical skill
                    if any(keyword in token.text.lower() for keyword in 
                          ['api', 'sql', 'js', 'ml', 'ai', 'dev', 'ops', 'web', 'app', 'data']):
                        skills.add(token.text)
            
            # Clean and normalize skills
            cleaned_skills = []
            for skill in skills:
                skill = skill.strip().lower()
                if len(skill) > 1 and skill not in ['api', 'js', 'ml', 'ai']:
                    cleaned_skills.append(skill)
            
            # Remove duplicates and sort
            cleaned_skills = list(set(cleaned_skills))
            cleaned_skills.sort()
            
            logger.info(f"Extracted {len(cleaned_skills)} potential skills")
            return cleaned_skills
            
        except Exception as e:
            logger.error(f"Error extracting skills: {str(e)}")
            raise
    
    def extract_education(self, text: str) -> List[str]:
        """
        Extract education information from text
        
        Args:
            text (str): Text to extract education from
            
        Returns:
            List[str]: List of education-related information
        """
        try:
            logger.info("Extracting education information...")
            
            # Education patterns
            education_patterns = [
                r'\b(bachelor|master|phd|doctorate|b\.?s\.?|m\.?s\.?|ph\.?d\.?)\b',
                r'\b(computer science|engineering|mathematics|statistics|data science|business|economics)\b',
                r'\b(university|college|institute|school)\b',
                r'\b(degree|certificate|certification|diploma)\b',
                r'\b(gpa|grade point average)\b'
            ]
            
            education_info = set()
            
            # Extract using patterns
            for pattern in education_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                education_info.update(matches)
            
            # Extract using spaCy NER for organizations (universities)
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'ORG':
                    if any(keyword in ent.text.lower() for keyword in 
                          ['university', 'college', 'institute', 'school']):
                        education_info.add(ent.text)
            
            education_list = list(education_info)
            education_list.sort()
            
            logger.info(f"Extracted {len(education_list)} education-related items")
            return education_list
            
        except Exception as e:
            logger.error(f"Error extracting education: {str(e)}")
            raise
    
    def extract_experience(self, text: str) -> List[str]:
        """
        Extract work experience information from text
        
        Args:
            text (str): Text to extract experience from
            
        Returns:
            List[str]: List of experience-related information
        """
        try:
            logger.info("Extracting experience information...")
            
            # Experience patterns
            experience_patterns = [
                r'\b(experience|exp|years?|months?|intern|internship|job|work|position|role)\b',
                r'\b(developer|engineer|analyst|manager|director|lead|senior|junior)\b',
                r'\b(company|corporation|inc|llc|ltd|startup|enterprise)\b',
                r'\b(project|team|department|division)\b'
            ]
            
            experience_info = set()
            
            # Extract using patterns
            for pattern in experience_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                experience_info.update(matches)
            
            # Extract using spaCy NER for organizations (companies)
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'ORG':
                    if any(keyword in ent.text.lower() for keyword in 
                          ['inc', 'corp', 'llc', 'ltd', 'company', 'technologies', 'solutions']):
                        experience_info.add(ent.text)
            
            experience_list = list(experience_info)
            experience_list.sort()
            
            logger.info(f"Extracted {len(experience_list)} experience-related items")
            return experience_list
            
        except Exception as e:
            logger.error(f"Error extracting experience: {str(e)}")
            raise
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize text using spaCy
        
        Args:
            text (str): Text to process
            
        Returns:
            List[str]: List of lemmatized tokens
        """
        try:
            logger.info("Tokenizing and lemmatizing text...")
            
            doc = self.nlp(text)
            tokens = []
            
            for token in doc:
                # Skip stop words, punctuation, and whitespace
                if not token.is_stop and not token.is_punct and not token.is_space:
                    # Get lemmatized form
                    lemma = token.lemma_.lower().strip()
                    if len(lemma) > 1:  # Skip single characters
                        tokens.append(lemma)
            
            logger.info(f"Tokenized and lemmatized {len(tokens)} tokens")
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing and lemmatizing: {str(e)}")
            raise
    
    def preprocess_resume(self, text: str) -> Dict[str, any]:
        """
        Complete preprocessing pipeline for resume text
        
        Args:
            text (str): Raw resume text
            
        Returns:
            Dict[str, any]: Dictionary containing all extracted information
        """
        try:
            logger.info("Starting resume preprocessing pipeline...")
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Extract entities
            entities = self.extract_entities(cleaned_text)
            
            # Extract specific information
            skills = self.extract_skills(cleaned_text)
            education = self.extract_education(cleaned_text)
            experience = self.extract_experience(cleaned_text)
            
            # Tokenize and lemmatize
            tokens = self.tokenize_and_lemmatize(cleaned_text)
            
            # Create result dictionary
            result = {
                'cleaned_text': cleaned_text,
                'tokens': tokens,
                'entities': entities,
                'skills': skills,
                'education': education,
                'experience': experience,
                'text_length': len(cleaned_text),
                'token_count': len(tokens)
            }
            
            logger.info("Resume preprocessing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in resume preprocessing: {str(e)}")
            raise
    
    def preprocess_job_description(self, text: str) -> Dict[str, any]:
        """
        Complete preprocessing pipeline for job description text
        
        Args:
            text (str): Raw job description text
            
        Returns:
            Dict[str, any]: Dictionary containing all extracted information
        """
        try:
            logger.info("Starting job description preprocessing pipeline...")
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Extract entities
            entities = self.extract_entities(cleaned_text)
            
            # Extract specific information
            skills = self.extract_skills(cleaned_text)
            education = self.extract_education(cleaned_text)
            experience = self.extract_experience(cleaned_text)
            
            # Tokenize and lemmatize
            tokens = self.tokenize_and_lemmatize(cleaned_text)
            
            # Create result dictionary
            result = {
                'cleaned_text': cleaned_text,
                'tokens': tokens,
                'entities': entities,
                'skills': skills,
                'education': education,
                'experience': experience,
                'text_length': len(cleaned_text),
                'token_count': len(tokens)
            }
            
            logger.info("Job description preprocessing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in job description preprocessing: {str(e)}")
            raise


# Convenience functions for easy usage
def preprocess_resume_text(text: str) -> Dict[str, any]:
    """
    Convenience function to preprocess resume text
    
    Args:
        text (str): Raw resume text
        
    Returns:
        Dict[str, any]: Preprocessed resume data
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_resume(text)


def preprocess_job_description_text(text: str) -> Dict[str, any]:
    """
    Convenience function to preprocess job description text
    
    Args:
        text (str): Raw job description text
        
    Returns:
        Dict[str, any]: Preprocessed job description data
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_job_description(text)


if __name__ == "__main__":
    # Test the text preprocessor
    preprocessor = TextPreprocessor()
    
    # Example usage
    sample_text = """
    John Smith
    Software Engineer
    Experience: 3 years in Python, JavaScript, and web development
    Education: Bachelor's in Computer Science from MIT
    Skills: Machine Learning, Data Science, React, Node.js
    """
    
    print("ResuMatch AI - Text Preprocessor")
    print("Testing preprocessing pipeline...")
    
    result = preprocessor.preprocess_resume(sample_text)
    print(f"Extracted skills: {result['skills']}")
    print(f"Extracted education: {result['education']}")
    print(f"Extracted experience: {result['experience']}")
