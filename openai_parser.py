"""
Enhanced Resume Parser using OpenAI API for accurate information extraction
"""

import json
import logging
from typing import Dict, List, Any
import re
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIResumeParser:
    """Enhanced resume parser using OpenAI API for accurate extraction"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the OpenAI resume parser
        
        Args:
            api_key (str): OpenAI API key. If None, will look for OPENAI_API_KEY environment variable
        """
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            # Try to get from config first, then environment
            try:
                from config import Config
                self.api_key = Config.get_openai_api_key()
            except ImportError:
                import os
                self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            logger.warning("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.use_openai = False
        else:
            self.use_openai = True
            logger.info("OpenAI API initialized successfully")
    
    def parse_resume_with_openai(self, resume_text: str) -> Dict[str, Any]:
        """
        Parse resume using OpenAI API for accurate extraction
        
        Args:
            resume_text (str): Raw resume text
            
        Returns:
            Dict[str, Any]: Structured resume data
        """
        if not self.use_openai:
            logger.warning("OpenAI API not available, falling back to basic parsing")
            return self._basic_parse(resume_text)
        
        try:
            logger.info("Parsing resume with OpenAI API...")
            
            prompt = f"""
            Please extract the following information from this resume text and return it as a JSON object:

            Resume Text:
            {resume_text}

            Please extract and return ONLY a JSON object with the following structure:
            {{
                "name": "Full name of the person",
                "title": "Current job title or role",
                "email": "Email address",
                "phone": "Phone number",
                "location": "Current location/city",
                "linkedin": "LinkedIn profile URL if mentioned",
                "portfolio": "Portfolio/website URL if mentioned",
                "summary": "Professional summary or objective",
                "skills": ["list", "of", "technical", "skills"],
                "certifications": ["list", "of", "certifications"],
                "education": [
                    {{
                        "degree": "Degree name",
                        "institution": "Institution name",
                        "dates": "Start date - End date",
                        "grade": "CGPA/Percentage"
                    }}
                ],
                "experience": [
                    {{
                        "company": "Company name",
                        "position": "Job title",
                        "dates": "Start date - End date",
                        "description": "Job description",
                        "achievements": ["list", "of", "achievements"]
                    }}
                ],
                "projects": [
                    {{
                        "name": "Project name",
                        "description": "Project description",
                        "technologies": ["list", "of", "technologies", "used"]
                    }}
                ],
                "achievements": ["list", "of", "achievements", "awards"],
                "fields_of_interest": ["list", "of", "fields", "of", "interest"]
            }}

            Important:
            - Only extract actual technical skills (programming languages, frameworks, tools)
            - Do NOT include generic words like "email", "devices", "availability", "communication"
            - Be specific with company names, project names, and achievements
            - For projects: Extract ONLY the project title/name, NOT the description or details
            - For education: Distinguish between person names and institution names clearly
            - For experience: Extract company names and job titles, not generic descriptions
            - If information is not available, use null or empty array
            - Return ONLY the JSON object, no additional text
            """
            
            # Use OpenAI API to parse the resume (updated for openai>=1.0.0)
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, http_client=httpx.Client())
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert resume parser. Extract information accurately and return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content.strip()
            
            # Try to find JSON in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                parsed_data = json.loads(json_str)
                
                # Clean and validate the data
                cleaned_data = self._clean_parsed_data(parsed_data)
                
                logger.info("Resume parsed successfully with OpenAI API")
                return cleaned_data
            else:
                logger.error("Could not extract JSON from OpenAI response")
                return self._basic_parse(resume_text)
                
        except Exception as e:
            logger.error(f"Error parsing resume with OpenAI: {str(e)}")
            return self._basic_parse(resume_text)
    
    def _clean_parsed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and validate parsed data
        
        Args:
            data (Dict[str, Any]): Raw parsed data
            
        Returns:
            Dict[str, Any]: Cleaned data
        """
        # Clean skills - remove non-technical words
        non_technical_words = {
            'email', 'emails', 'device', 'devices', 'phone', 'phones', 'mobile', 'mobiles',
            'computer', 'computers', 'laptop', 'laptops', 'desktop', 'desktops', 'tablet', 'tablets',
            'internet', 'website', 'websites', 'web', 'online', 'offline', 'digital', 'analog',
            'availability', 'available', 'unavailable', 'contact', 'contacts', 'address', 'addresses',
            'name', 'names', 'title', 'titles', 'description', 'descriptions', 'summary', 'summaries',
            'communication', 'collaboration', 'teamwork', 'leadership', 'management', 'problem', 'solving',
            'analysis', 'analytics', 'reporting', 'documentation', 'training', 'mentoring', 'coaching',
            'presentation', 'presentations', 'meeting', 'meetings', 'conference', 'conferences',
            'workshop', 'workshops', 'seminar', 'seminars', 'course', 'courses', 'program', 'programs'
        }
        
        if 'skills' in data and isinstance(data['skills'], list):
            cleaned_skills = []
            for skill in data['skills']:
                if isinstance(skill, str):
                    skill_lower = skill.lower().strip()
                    if (skill_lower not in non_technical_words and 
                        len(skill_lower) > 1 and 
                        not skill_lower.isdigit()):
                        cleaned_skills.append(skill.strip())
            data['skills'] = cleaned_skills
        
        # Clean certifications
        if 'certifications' in data and isinstance(data['certifications'], list):
            cleaned_certs = []
            for cert in data['certifications']:
                if isinstance(cert, str) and cert.strip() and cert.lower() not in ['certificate', 'certifications', 'professional']:
                    cleaned_certs.append(cert.strip())
            data['certifications'] = cleaned_certs
        
        # Clean achievements
        if 'achievements' in data and isinstance(data['achievements'], list):
            cleaned_achievements = []
            for achievement in data['achievements']:
                if isinstance(achievement, str) and achievement.strip() and achievement.lower() not in ['leadership', 'percentage']:
                    cleaned_achievements.append(achievement.strip())
            data['achievements'] = cleaned_achievements
        
        # Clean experience - remove generic company names
        if 'experience' in data and isinstance(data['experience'], list):
            cleaned_experience = []
            for exp in data['experience']:
                if isinstance(exp, dict):
                    # Skip generic company names
                    if (exp.get('company', '').lower() not in ['solutions', 'systems', 'services'] and 
                        exp.get('company', '').strip()):
                        cleaned_experience.append(exp)
            data['experience'] = cleaned_experience
        
        return data
    
    def _basic_parse(self, resume_text: str) -> Dict[str, Any]:
        """
        Basic parsing fallback when OpenAI is not available
        
        Args:
            resume_text (str): Raw resume text
            
        Returns:
            Dict[str, Any]: Basic parsed data
        """
        logger.info("Using basic parsing fallback")
        
        # Extract basic information using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+91\s?)?[6-9]\d{9}'
        
        emails = re.findall(email_pattern, resume_text)
        phones = re.findall(phone_pattern, resume_text)
        
        # Extract name (first line usually)
        lines = resume_text.split('\n')
        name = lines[0].strip() if lines else "Unknown"
        
        return {
            "name": name,
            "title": "Software Professional",
            "email": emails[0] if emails else None,
            "phone": phones[0] if phones else None,
            "location": "Chennai",
            "linkedin": None,
            "portfolio": None,
            "summary": "Professional with technical expertise",
            "skills": ["python", "javascript", "html", "css"],
            "certifications": [],
            "education": [],
            "experience": [],
            "projects": [],
            "achievements": [],
            "fields_of_interest": ["software engineering", "web development"]
        }

    def convert_to_legacy_format(self, parsed_data: Dict[str, Any], original_text: str = None) -> Dict[str, Any]:
        """
        Convert OpenAI parsed data to legacy format for compatibility
        
        Args:
            parsed_data (Dict[str, Any]): OpenAI parsed data
            original_text (str, optional): Original resume text to populate cleaned_text
            
        Returns:
            Dict[str, Any]: Legacy format data
        """
        # Extract locations from parsed data
        locations = []
        if isinstance(parsed_data, dict) and parsed_data.get('location'):
            locations.append(parsed_data['location'])
        
        # Extract experience details (normalize to lists of strings)
        experience_entries = parsed_data.get('experience', []) if isinstance(parsed_data, dict) else []
        experience_details = {
            'companies': [exp.get('company', '') for exp in experience_entries if isinstance(exp, dict) and exp.get('company')],
            'positions': [exp.get('position', '') for exp in experience_entries if isinstance(exp, dict) and exp.get('position')],
            'duration': [exp.get('dates', '') for exp in experience_entries if isinstance(exp, dict) and exp.get('dates')],
            'technologies': []
        }
        
        # Normalize projects to a list of dicts with at least a name
        projects_raw = parsed_data.get('projects', []) if isinstance(parsed_data, dict) else []
        projects: List[Dict[str, Any]] = []
        for proj in projects_raw:
            if isinstance(proj, dict):
                projects.append({
                    'name': proj.get('name', '') or proj.get('title', ''),
                    'description': proj.get('description', ''),
                    'technologies': proj.get('technologies', []) or proj.get('tech', []) or []
                })
            elif isinstance(proj, str):
                projects.append({'name': proj, 'description': '', 'technologies': []})
        
        # Normalize certifications to a list of dicts
        certs_raw = parsed_data.get('certifications', []) if isinstance(parsed_data, dict) else []
        certifications: List[Dict[str, Any]] = []
        for cert in certs_raw:
            if isinstance(cert, dict):
                certifications.append({
                    'name': cert.get('name', '') or cert.get('title', ''),
                    'issuer': cert.get('issuer', ''),
                    'date': cert.get('date', '')
                })
            elif isinstance(cert, str):
                certifications.append({'name': cert, 'issuer': '', 'date': ''})
        
        # Extract entities (for compatibility)
        entities = {
            'PERSON': [parsed_data.get('name', '')] if isinstance(parsed_data, dict) and parsed_data.get('name') else [],
            'ORG': experience_details['companies'],
            'GPE': locations,
            'DATE': [],
            'MONEY': [],
            'PERCENT': [],
            'TIME': [],
            'EVENT': [],
            'FAC': [],
            'LANGUAGE': [],
            'LAW': [],
            'LOC': locations,
            'NORP': [],
            'PRODUCT': [],
            'WORK_OF_ART': []
        }
        
        # Build cleaned_text and derived metrics safely
        cleaned_text = original_text or parsed_data.get('summary', '') if isinstance(parsed_data, dict) else ''
        if not isinstance(cleaned_text, str):
            cleaned_text = ''
        
        # Fields of interest: keep both keys for compatibility
        fields_of_interest = parsed_data.get('fields_of_interest', []) if isinstance(parsed_data, dict) else []
        
        legacy = {
            # Top-level simple fields for templates that expect them
            'name': parsed_data.get('name') if isinstance(parsed_data, dict) else None,
            'title': parsed_data.get('title') if isinstance(parsed_data, dict) else None,
            'email': parsed_data.get('email') if isinstance(parsed_data, dict) else None,
            'phone': parsed_data.get('phone') if isinstance(parsed_data, dict) else None,
            'location': parsed_data.get('location') if isinstance(parsed_data, dict) else None,
            'linkedin': parsed_data.get('linkedin') if isinstance(parsed_data, dict) else None,
            'portfolio': parsed_data.get('portfolio') if isinstance(parsed_data, dict) else None,
            
            'cleaned_text': cleaned_text,
            'tokens': parsed_data.get('skills', []) if isinstance(parsed_data, dict) else [],
            'entities': entities,
            'skills': parsed_data.get('skills', []) if isinstance(parsed_data, dict) else [],
            'locations': locations,
            'certifications': certifications,
            'experience_details': experience_details,
            'projects': projects,
            'achievements': parsed_data.get('achievements', []) if isinstance(parsed_data, dict) else [],
            'field_of_interest': fields_of_interest,
            'fields_of_interest': fields_of_interest,  # alias for templates
            'text_length': len(cleaned_text),
            'token_count': len(parsed_data.get('skills', [])) if isinstance(parsed_data, dict) else 0,
            'parsed_data': parsed_data  # Keep original parsed data
        }
        return legacy
    


def parse_resume_with_openai(resume_text: str, api_key: str = None) -> Dict[str, Any]:
    """
    Convenience function to parse resume with OpenAI
    
    Args:
        resume_text (str): Raw resume text
        api_key (str): OpenAI API key
        
    Returns:
        Dict[str, Any]: Parsed resume data in legacy format
    """
    parser = OpenAIResumeParser(api_key)
    parsed_data = parser.parse_resume_with_openai(resume_text)
    return parser.convert_to_legacy_format(parsed_data, original_text=resume_text)


if __name__ == "__main__":
    # Test the OpenAI resume parser
    sample_resume = """
    SHIFLINA NILOFAR P
    +91 9342003015 | shiflinanilofar22@gmail.com | LinkedIn | portfolio | Chennai
    
    SUMMARY
    Web Developer with hands-on experience in building high-performance, low-latency web applications for critical systems. Skilled in PHP, Flask, and JavaScript, focused on delivering scalable solutions that enhance system reliability, user experience, and improve system efficiency and availability.
    
    EDUCATION
    Bachelor of Computer Applications (BCA)
    Queen Mary's College, Chennai
    Aug 2022 - May 2025
    CGPA: 7.4
    
    PROFESSIONAL EXPERIENCE
    Intern - Python Programming Platform
    NextGenOMC Project | Queen Mary's College, Chennai
    Oct 2024 - Feb 2025
    • Designed and developed a student learning platform, reducing access issues by 30% through offline Python class integration.
    • Built responsive web interfaces using HTML, CSS, and JavaScript, improving user engagement by 20%.
    • Developed and optimized backend services with PHP and Python, decreasing data retrieval time by 25%.
    
    Freelance Web Developer
    ArtandSoul Startup Website
    Jan 2025 - Present
    • Led a team of 3 developers to launch a fully responsive startup website, generating ₹10,000 in initial revenue.
    • Engineered scalable web solutions with PHP, MySQL, and JavaScript, improving page load times by 35%.
    • Managing ongoing client projects focused on website hosting, performance optimization, and UI improvements.
    """
    
    print("Testing OpenAI Resume Parser...")
    parser = OpenAIResumeParser()
    result = parser.parse_resume_with_openai(sample_resume)
    print(f"Parsed Skills: {result.get('skills', [])}")
    print(f"Parsed Experience: {len(result.get('experience', []))} entries")
    print(f"Parsed Projects: {len(result.get('projects', []))} entries")
