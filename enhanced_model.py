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
import random
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
                
                # Heuristics + seeded randomization for experience metadata if not present
                jd_low = (job.get('description','') + ' ' + job.get('title','') + ' ' + ' '.join(job.get('requirements', []))).lower()
                seeded = random.Random(hash(job.get('id')) if job.get('id') is not None else 0)
                exp_level = job.get('experience_level') or ('intern' if 'intern' in jd_low else seeded.choice(['fresher','junior']))
                if isinstance(job.get('min_experience_years'), (int, float)):
                    min_years = float(job.get('min_experience_years'))
                else:
                    if exp_level == 'intern':
                        # avoid zero: require at least 0.5 year for interns
                        min_years = seeded.choice([0.5, 1.0])
                    elif exp_level == 'fresher':
                        min_years = seeded.choice([0.5, 1.0])
                    else:  # junior
                        min_years = seeded.choice([1.0, 1.5, 2.0])
                if isinstance(job.get('min_internships'), int):
                    min_internships = int(job.get('min_internships'))
                else:
                    min_internships = seeded.choice([0, 1, 2])
                if isinstance(job.get('min_programs'), int):
                    min_programs = int(job.get('min_programs'))
                else:
                    min_programs = seeded.choice([0, 1])
                # Ensure not all zeros
                if (min_years == 0.0 and min_internships == 0 and min_programs == 0):
                    min_internships = 1

                # Normalize field_of_interest to a list
                job_field = job.get('field', '')
                job_foi = job.get('field_of_interest')
                if isinstance(job_foi, str):
                    job_foi = [job_foi]
                if not isinstance(job_foi, list):
                    job_foi = [job_field] if job_field else []

                job_info = {
                    'id': job.get('id'),
                    'title': job.get('title', ''),
                    'company': job.get('company', ''),
                    'description': job.get('description', ''),
                    'requirements': job.get('requirements', []),
                    'skills_needed': job.get('skills_needed', []),
                    'location': job.get('location', ''),
                    'duration': job.get('duration', ''),
                    'field': job_field,
                    'field_of_interest': job_foi,
                    'experience_level': exp_level,
                    'min_experience_years': float(min_years),
                    'min_internships': int(min_internships),
                    'min_programs': int(min_programs),
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
            if not job_location:
                return 0.5  # Neutral if job has no location
            # Normalize inputs
            job_location_lower = (job_location or '').lower()
            resume_locations_lower = [str(loc).lower() for loc in (resume_locations or []) if isinstance(loc, str)]
            # If resume didn't list locations, try to parse from a single string (e.g., "Pune, India")
            # This function expects caller to pass any top-level 'location' as well.

            if not resume_locations_lower:
                return 0.4  # Slightly below neutral when candidate gave nothing

            # Exact/substring match
            for resume_loc in resume_locations_lower:
                if resume_loc and (resume_loc in job_location_lower or job_location_lower in resume_loc):
                    return 1.0

            # City normalization and synonyms
            synonyms = {
                'bengaluru': 'bangalore',
                'gurugram': 'gurgaon',
                'mysuru': 'mysore',
                'kozhikode': 'calicut'
            }
            def normalize_city(s: str) -> str:
                s = s.strip().lower()
                for k, v in synonyms.items():
                    s = s.replace(k, v)
                return s

            job_city = None
            jl_norm = normalize_city(job_location_lower)
            for city in self.indian_cities:
                if city in jl_norm:
                    job_city = city
                    break

            if job_city:
                for resume_loc in resume_locations_lower:
                    rl = normalize_city(resume_loc)
                    if job_city in rl:
                        return 0.85  # High score for same city

            # Country match (India)
            if 'india' in jl_norm:
                for resume_loc in resume_locations_lower:
                    if 'india' in normalize_city(resume_loc):
                        return 0.6  # Medium score for same country

            return 0.25  # Mismatch

        except Exception as e:
            logger.error(f"Error calculating location score: {str(e)}")
            return 0.4
    
    def calculate_skill_match_score(self, resume_skills: List[str], job_skills: List[str]) -> Tuple[float, List[str], List[str]]:
        """
        Calculate skill match score and identify missing skills
        
        Args:
            resume_skills (List[str]): Skills from resume
            job_skills (List[str]): Skills required for job (augmented list)
            
        Returns:
            Tuple[float, List[str], List[str]]: (match_score, matched_skills, missing_skills)
        """
        try:
            resume_skills = [s for s in (resume_skills or []) if isinstance(s, str) and s.strip()]
            job_skills = [s for s in (job_skills or []) if isinstance(s, str) and s.strip()]
            if not job_skills:
                return 0.0, [], []  # No declared skills in job -> don't inflate score

            resume_skills_lower = [skill.lower().strip() for skill in resume_skills]
            job_skills_lower = [skill.lower().strip() for skill in job_skills]

            matched_skills = []
            missing_skills = []

            for job_skill in job_skills_lower:
                found = False
                for resume_skill in resume_skills_lower:
                    if (job_skill in resume_skill or resume_skill in job_skill or self.are_skills_similar(job_skill, resume_skill)):
                        matched_skills.append(job_skill)
                        found = True
                        break
                if not found:
                    missing_skills.append(job_skill)

            # Strict ratio based on job requirements
            match_score = len(matched_skills) / max(1, len(job_skills_lower))

            # Slightly dampen scores with very small intersections
            if len(matched_skills) == 0:
                match_score = 0.0
            elif len(matched_skills) == 1 and len(job_skills_lower) >= 5:
                match_score = min(match_score, 0.2)

            # De-duplicate for output (restore original casing best-effort)
            matched_uniq = sorted(list({m for m in matched_skills}))
            missing_uniq = sorted(list({m for m in missing_skills}))

            return match_score, matched_uniq, missing_uniq

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
    
    def _calculate_experience_match(self, resume_data: Dict[str, Any], job: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Compute experience match score based on years, internships, and programs.
        Returns (score in [0,1], details dict)."""
        try:
            import re
            text = resume_data.get('cleaned_text') or ''
            # Years from text (sum up simple 'X years' mentions, cap at 10)
            years = 0.0
            for m in re.findall(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", text, flags=re.I):
                try:
                    years += float(m)
                except Exception:
                    pass
            years = min(years, 10.0)
            # Internships from structured experience and text
            internships = 0
            for e in (resume_data.get('experience') or []):
                if isinstance(e, dict) and 'position' in e and isinstance(e['position'], str) and ('intern' in e['position'].lower()):
                    internships += 1
            if internships == 0:
                internships += len(re.findall(r"\bintern\b", text, flags=re.I))
            # Programs participated: count hints in achievements/certifications
            programs = 0
            for a in (resume_data.get('achievements') or []):
                if isinstance(a, str) and any(k in a.lower() for k in ['program', 'simulation', 'virtual experience']):
                    programs += 1
            for c in (resume_data.get('certifications') or []):
                if isinstance(c, dict):
                    nm = (c.get('name') or '') + ' ' + (c.get('issuer') or '')
                    if any(k in nm.lower() for k in ['program', 'simulation', 'virtual experience']):
                        programs += 1
                elif isinstance(c, str) and any(k in c.lower() for k in ['program','simulation','virtual experience']):
                    programs += 1

            # Default 6 months for each internship/program if duration not given explicitly
            derived_years = min(10.0, years + 0.5 * internships + 0.5 * programs)
            # Count distinct companies in structured experience
            company_count = 0
            try:
                companies = []
                for e in (resume_data.get('experience') or []):
                    if isinstance(e, dict):
                        comp = (e.get('company') or '').strip()
                        if comp:
                            companies.append(comp.lower())
                company_count = len(set(companies))
            except Exception:
                company_count = 0

            req_years = float(job.get('min_experience_years') or 0.0)
            req_intern = int(job.get('min_internships') or 0)
            req_prog = int(job.get('min_programs') or 0)

            parts = []
            # If job specifies requirement, compute ratio; else treat as satisfied
            if req_years > 0:
                parts.append(min(1.0, derived_years / req_years))
            if req_intern > 0:
                parts.append(min(1.0, internships / req_intern))
            if req_prog > 0:
                parts.append(min(1.0, programs / req_prog))

            score = 1.0 if not parts else float(sum(parts) / len(parts))
            details = {
                'resume_years_explicit': years,
                'resume_years_total': derived_years,
                'resume_internships': internships,
                'resume_programs': programs,
                'resume_company_count': company_count,
                'required_years': req_years,
                'required_internships': req_intern,
                'required_programs': req_prog
            }
            return score, details
        except Exception as e:
            logger.warning(f"Experience match failed: {e}")
            return 0.5, {'error': str(e)}

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
            
            resume_text = resume_data.get('cleaned_text', '')
            resume_skills = resume_data.get('skills', []) or []
            # Merge top-level 'location' into locations list if present
            resume_locations = list({
                *(resume_data.get('locations', []) or []),
                *( [resume_data.get('location')] if resume_data.get('location') else [] )
            })
            resume_field = resume_data.get('fields_of_interest') or resume_data.get('field_of_interest') or []
            locks = set([str(s).lower() for s in (resume_data.get('locked_sections') or []) if isinstance(s, str)])

            # Get basic similarity scores (top-K*2 to seed but also evaluate all jobs later)
            tfidf_results = self.predict_tfidf(resume_text, top_k * 2)
            sbert_results = self.predict_sentence_transformer(resume_text, top_k * 2)
            
            # Create lookup for quick retrieval
            tfidf_map = {r['job_id']: r['similarity_score'] for r in tfidf_results}
            sbert_map = {r['job_id']: r['similarity_score'] for r in sbert_results}

            enhanced_results = []
            
            for job in self.job_descriptions:
                # Base semantic similarities (default 0 if not in the top-k seed)
                tfidf_score = float(tfidf_map.get(job['id'], 0.0))
                sbert_score = float(sbert_map.get(job['id'], 0.0))

                # Derive/augment job skills from requirements/description if needed
                job_skills = [s for s in (job.get('skills_needed') or []) if isinstance(s, str)]
                if len(job_skills) < 5:
                    import re
                    tokens = []
                    for r in (job.get('requirements') or []):
                        if isinstance(r, str):
                            tokens += re.findall(r'[A-Za-z][A-Za-z0-9\+\.#\-]{2,}', r)
                    tokens += re.findall(r'[A-Za-z][A-Za-z0-9\+\.#\-]{2,}', job.get('description') or '')
                    # keep plausible skill-like tokens
                    keep = []
                    for t in tokens:
                        tl = t.lower()
                        if any(x in tl for x in ['python','java','sql','node','react','django','flask','aws','azure','gcp','docker','kubernetes','ml','nlp','api','graphql','mongodb','mysql','postgres','android','ios','devops','linux','bash']):
                            keep.append(tl)
                    job_skills = sorted(list(dict.fromkeys(job_skills + keep)))

                # Calculate location score
                location_score = self.calculate_location_score(resume_locations, job.get('location', ''))
                
                # Calculate skill match
                skill_match_score, matched_skills, missing_skills = self.calculate_skill_match_score(
                    resume_skills, job_skills
                )

                # Experience match
                experience_match_score, experience_details = self._calculate_experience_match(resume_data, job)
                
                # Calculate field match using job field_of_interest vs resume fields_of_interest
                job_foi = job.get('field_of_interest') or []
                field_match_score = 0.0
                if isinstance(resume_field, list) and resume_field and isinstance(job_foi, list) and job_foi:
                    rf = [str(x).lower() for x in resume_field if isinstance(x, str)]
                    jf = [str(x).lower() for x in job_foi if isinstance(x, str)]
                    # Overlap if any substring match either way
                    match = False
                    for a in rf:
                        for b in jf:
                            if a in b or b in a:
                                match = True
                                break
                        if match:
                            break
                    field_match_score = 1.0 if match else 0.0
                else:
                    field_match_score = 0.0

                # Re-weight with stronger emphasis on skills, reduce semantic inflation (add experience)
                w_tfidf, w_sbert, w_skill, w_location, w_field, w_exp = 0.15, 0.20, 0.40, 0.15, 0.05, 0.05
                base_score = (
                    tfidf_score * w_tfidf +
                    sbert_score * w_sbert +
                    skill_match_score * w_skill +
                    location_score * w_location +
                    field_match_score * w_field +
                    experience_match_score * w_exp
                )

                # Gating/penalties to avoid fake high scores
                penalties = {
                    'skill_gate': 1.0,
                    'location_gate': 1.0,
                    'field_penalty': 0.0
                }
                # If user explicitly edited skills, enforce skill gate more strongly
                skill_gate_threshold = 0.2
                if ('skills' in locks) and skill_match_score < 0.2:
                    penalties['skill_gate'] = 0.7
                elif skill_match_score == 0.0:
                    penalties['skill_gate'] = 0.7
                elif skill_match_score < skill_gate_threshold:
                    penalties['skill_gate'] = 0.85

                # If user edited location and it's a mismatch, penalize more
                if ('location' in locks) and location_score <= 0.3:
                    penalties['location_gate'] = 0.8
                elif location_score <= 0.3 and resume_locations:
                    penalties['location_gate'] = 0.9

                # If fields of interest exist and don't match, small penalty
                if resume_field and field_match_score == 0.0:
                    penalties['field_penalty'] = 0.03

                combined_score = max(0.0, base_score * penalties['skill_gate'] * penalties['location_gate'] - penalties['field_penalty'])

                # If all key dimensions match, force 100%
                all_matched = (
                    skill_match_score >= 0.999 and
                    location_score >= 0.85 and
                    field_match_score == 1.0 and
                    experience_match_score >= 0.999
                )
                if all_matched:
                    combined_score = 1.0

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
                    'field_of_interest': job.get('field_of_interest', []),
                    'experience_level': job.get('experience_level'),
                    'min_experience_years': job.get('min_experience_years'),
                    'min_internships': job.get('min_internships'),
                    'min_programs': job.get('min_programs'),
                    'similarity_score': combined_score,
                    'similarity_percentage': combined_score * 100,
                    'method': 'Enhanced',
                    'breakdown': {
                        'tfidf_score': tfidf_score,
                        'sbert_score': sbert_score,
                        'location_score': location_score,
                        'skill_match_score': skill_match_score,
                        'field_match_score': field_match_score,
                        'experience_match_score': experience_match_score,
                        'weights': {
                            'tfidf': w_tfidf,
                            'sbert': w_sbert,
                            'skills': w_skill,
                            'location': w_location,
                            'field': w_field,
                            'experience': w_exp
                        },
                        'penalties': penalties
                    },
                    'skill_analysis': {
                        'matched_skills': matched_skills,
                        'missing_skills': missing_skills,
                        'skill_match_percentage': skill_match_score * 100
                    },
                    'experience_analysis': experience_details,
                    'learning_suggestions': learning_suggestions,
                    'location_preference': location_score > 0.7
                }
                
                enhanced_results.append(enhanced_result)
            
            # Sort by combined score and return top-k
            enhanced_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            enhanced_results = enhanced_results[:top_k]
            
            top_pct = enhanced_results[0]['similarity_percentage'] if enhanced_results else 0.0
            logger.info(f"Enhanced prediction completed. Top similarity: {top_pct:.2f}%")
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
