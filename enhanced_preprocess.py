"""
Enhanced NLP preprocessing module for ResuMatch AI
Handles text cleaning, tokenization, lemmatization, and Named Entity Recognition
with improved location understanding and detailed resume extraction
"""

import spacy
import re
import logging
from typing import List, Dict, Set, Tuple, Any
import string
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTextPreprocessor:
    """Enhanced text preprocessor with better location understanding and detailed extraction"""
    
    def __init__(self):
        """Initialize the enhanced text preprocessor with spaCy model"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded spaCy model: en_core_web_sm")
            
            # Canonical section headings (lowercase)
            self.section_aliases = {
                'education': {'education', 'educational background', 'academics'},
                'projects': {'projects', 'project work', 'project', 'academic projects'},
                'experience': {'experience', 'work experience', 'professional experience'},
                'skills': {'skills', 'technical skills', 'core skills', 'languages & tools', 'languages and tools'},
                'certifications': {
                    'certifications', 'certification', 'licenses & certifications', 'licenses and certifications',
                    'simulation', 'simulations', 'job simulation', 'virtual experience', 'virtual experience program'
                },
                'achievements': {'achievements', 'awards', 'honors'},
                'summary': {'summary', 'professional summary', 'profile', 'objective'}
            }
            
            # Define Indian cities and locations
            self.indian_cities = {
                'bangalore', 'bengaluru', 'mumbai', 'delhi', 'hyderabad', 'chennai', 'pune', 'kolkata', 'ahmedabad',
                'gurgaon', 'gurugram', 'noida', 'faridabad', 'ghaziabad', 'jaipur', 'lucknow', 'kanpur', 'nagpur',
                'indore', 'bhopal', 'visakhapatnam', 'vijayawada', 'coimbatore', 'madurai', 'salem', 'tiruchirappalli',
                'kochi', 'thiruvananthapuram', 'calicut', 'kozhikode', 'mysore', 'mysuru', 'mangalore', 'hubli',
                'belgaum', 'gulbarga', 'kalaburagi', 'raichur', 'bidar', 'bijapur', 'vijayapura', 'bagalkot',
                'kolar', 'tumkur', 'chitradurga', 'shimoga', 'udupi', 'dakshina kannada', 'udupi', 'hassan',
                'mandya', 'chamrajanagar', 'kodagu', 'chikkamagaluru', 'chikmagalur', 'davangere', 'bellary',
                'ballari', 'kurnool', 'anantapur', 'kadapa', 'chittoor', 'nellore', 'ongole', 'guntur', 'vijayawada',
                'rajahmundry', 'kakinada', 'eluru', 'bhimavaram', 'tadepalligudem', 'tenali', 'chilakaluripet',
                'narasaraopet', 'palakollu', 'tadipatri', 'proddatur', 'kadiri', 'puttur', 'srikakulam', 'vizianagaram'
            }
            
            # Define common non-skill words that might be confused as skills
            self.non_skill_words = {
                'application', 'applications', 'apply', 'applied', 'applying', 'applicant', 'applicants',
                'chennai', 'bangalore', 'mumbai', 'delhi', 'hyderabad', 'pune', 'kolkata', 'ahmedabad',
                'gurgaon', 'noida', 'india', 'usa', 'united states', 'america', 'canada', 'uk', 'united kingdom',
                'experience', 'experiences', 'experienced', 'experiencing', 'work', 'works', 'working', 'worked',
                'project', 'projects', 'projected', 'projecting', 'certification', 'certifications', 'certified',
                'certifying', 'achievement', 'achievements', 'achieved', 'achieving', 'education', 'educational',
                'studied', 'studying', 'study', 'studies', 'degree', 'degrees', 'diploma', 'diplomas',
                'bachelor', 'masters', 'phd', 'doctorate', 'university', 'college', 'institute', 'school',
                'company', 'companies', 'organization', 'organizations', 'corporation', 'corporations',
                'internship', 'internships', 'intern', 'interns', 'job', 'jobs', 'position', 'positions',
                'role', 'roles', 'responsibility', 'responsibilities', 'duty', 'duties', 'task', 'tasks',
                'email', 'emails', 'device', 'devices', 'phone', 'phones', 'mobile', 'mobiles', 'computer',
                'computers', 'laptop', 'laptops', 'desktop', 'desktops', 'tablet', 'tablets', 'internet',
                'website', 'websites', 'web', 'online', 'offline', 'digital', 'analog', 'electronic',
                'availability', 'available', 'unavailable', 'contact', 'contacts', 'address', 'addresses',
                'name', 'names', 'title', 'titles', 'description', 'descriptions', 'summary', 'summaries',
                'objective', 'objectives', 'goal', 'goals', 'target', 'targets', 'aim', 'aims', 'purpose',
                'purposes', 'mission', 'missions', 'vision', 'visions', 'value', 'values', 'principle', 'principles',
                'method', 'methods', 'approach', 'approaches', 'strategy', 'strategies', 'technique', 'techniques',
                'process', 'processes', 'procedure', 'procedures', 'system', 'systems', 'platform', 'platforms',
                'service', 'services', 'product', 'products', 'solution', 'solutions', 'tool', 'tools',
                'software', 'hardware', 'technology', 'technologies', 'innovation', 'innovations', 'research',
                'development', 'design', 'testing', 'implementation', 'deployment', 'maintenance', 'support',
                'management', 'leadership', 'communication', 'collaboration', 'teamwork', 'problem', 'solving',
                'analysis', 'analytics', 'reporting', 'documentation', 'training', 'mentoring', 'coaching',
                'presentation', 'presentations', 'meeting', 'meetings', 'conference', 'conferences', 'workshop',
                'workshops', 'seminar', 'seminars', 'course', 'courses', 'program', 'programs', 'curriculum',
                'curricula', 'subject', 'subjects', 'topic', 'topics', 'area', 'areas', 'field', 'fields',
                'domain', 'domains', 'industry', 'industries', 'sector', 'sectors', 'market', 'markets',
                'business', 'commerce', 'finance', 'banking', 'insurance', 'healthcare', 'medical', 'pharmaceutical',
                'retail', 'ecommerce', 'manufacturing', 'automotive', 'aerospace', 'defense', 'energy', 'utilities',
                'telecommunications', 'media', 'entertainment', 'gaming', 'sports', 'fitness', 'wellness',
                'travel', 'hospitality', 'tourism', 'food', 'beverage', 'restaurant', 'hotel', 'transportation',
                'logistics', 'supply', 'chain', 'distribution', 'warehouse', 'inventory', 'procurement',
                'purchasing', 'sales', 'marketing', 'advertising', 'promotion', 'branding', 'public', 'relations',
                'customer', 'service', 'support', 'help', 'desk', 'technical', 'support', 'user', 'experience',
                'interface', 'usability', 'accessibility', 'performance', 'optimization', 'scalability', 'reliability',
                'security', 'privacy', 'compliance', 'governance', 'risk', 'management', 'quality', 'assurance',
                'testing', 'validation', 'verification', 'debugging', 'troubleshooting', 'monitoring', 'logging',
                'backup', 'recovery', 'disaster', 'recovery', 'business', 'continuity', 'change', 'management',
                'project', 'management', 'agile', 'scrum', 'kanban', 'lean', 'six', 'sigma', 'itil', 'cobit',
                'pmp', 'prince2', 'waterfall', 'vmodel', 'spiral', 'prototype', 'rapid', 'application', 'development',
                'extreme', 'programming', 'test', 'driven', 'development', 'behavior', 'driven', 'development',
                'continuous', 'integration', 'continuous', 'deployment', 'devops', 'sre', 'site', 'reliability',
                'engineering', 'cloud', 'computing', 'virtualization', 'containerization', 'microservices',
                'serverless', 'edge', 'computing', 'iot', 'internet', 'of', 'things', 'ai', 'artificial',
                'intelligence', 'machine', 'learning', 'deep', 'learning', 'neural', 'networks', 'nlp',
                'natural', 'language', 'processing', 'computer', 'vision', 'robotics', 'automation', 'rpa',
                'robotic', 'process', 'automation', 'blockchain', 'cryptocurrency', 'bitcoin', 'ethereum',
                'smart', 'contracts', 'defi', 'decentralized', 'finance', 'nft', 'non', 'fungible', 'tokens',
                'ar', 'augmented', 'reality', 'vr', 'virtual', 'reality', 'metaverse', 'web3', 'semantic',
                'web', 'linked', 'data', 'knowledge', 'graphs', 'ontologies', 'taxonomies', 'folksonomies',
                'social', 'networking', 'collaboration', 'tools', 'enterprise', 'social', 'networks', 'wikis',
                'blogs', 'forums', 'chat', 'messaging', 'video', 'conferencing', 'webinars', 'podcasts',
                'streaming', 'content', 'management', 'digital', 'asset', 'management', 'document', 'management',
                'records', 'management', 'knowledge', 'management', 'information', 'management', 'data',
                'governance', 'master', 'data', 'management', 'data', 'quality', 'data', 'cleansing',
                'data', 'integration', 'etl', 'extract', 'transform', 'load', 'data', 'warehousing',
                'data', 'lakes', 'data', 'marts', 'olap', 'online', 'analytical', 'processing', 'olap',
                'online', 'transaction', 'processing', 'oltp', 'real', 'time', 'analytics', 'stream',
                'processing', 'batch', 'processing', 'event', 'driven', 'architecture', 'message', 'queues',
                'pub', 'sub', 'publish', 'subscribe', 'event', 'sourcing', 'cqrs', 'command', 'query',
                'responsibility', 'segregation', 'domain', 'driven', 'design', 'ddd', 'hexagonal', 'architecture',
                'clean', 'architecture', 'onion', 'architecture', 'layered', 'architecture', 'n', 'tier',
                'architecture', 'soa', 'service', 'oriented', 'architecture', 'rest', 'representational',
                'state', 'transfer', 'graphql', 'api', 'application', 'programming', 'interface', 'web',
                'services', 'soap', 'simple', 'object', 'access', 'protocol', 'xml', 'extensible', 'markup',
                'language', 'json', 'javascript', 'object', 'notation', 'yaml', 'yet', 'another', 'markup',
                'language', 'toml', 'tom', 'obvious', 'minimal', 'language', 'ini', 'configuration', 'files',
                'environment', 'variables', 'secrets', 'management', 'vault', 'consul', 'etcd', 'zookeeper',
                'service', 'discovery', 'load', 'balancing', 'reverse', 'proxy', 'cdn', 'content', 'delivery',
                'network', 'edge', 'caching', 'redis', 'memcached', 'elasticsearch', 'solr', 'apache', 'lucene',
                'full', 'text', 'search', 'faceted', 'search', 'fuzzy', 'search', 'autocomplete', 'suggestions',
                'recommendation', 'engines', 'collaborative', 'filtering', 'content', 'based', 'filtering',
                'hybrid', 'recommendation', 'systems', 'a', 'b', 'testing', 'multivariate', 'testing',
                'conversion', 'optimization', 'landing', 'page', 'optimization', 'seo', 'search', 'engine',
                'optimization', 'sem', 'search', 'engine', 'marketing', 'ppc', 'pay', 'per', 'click',
                'cpc', 'cost', 'per', 'click', 'cpm', 'cost', 'per', 'mille', 'cpa', 'cost', 'per',
                'acquisition', 'roi', 'return', 'on', 'investment', 'roas', 'return', 'on', 'ad', 'spend',
                'ltv', 'lifetime', 'value', 'cac', 'customer', 'acquisition', 'cost', 'churn', 'rate',
                'retention', 'rate', 'engagement', 'rate', 'click', 'through', 'rate', 'ctr', 'conversion',
                'rate', 'bounce', 'rate', 'session', 'duration', 'page', 'views', 'unique', 'visitors',
                'traffic', 'sources', 'organic', 'search', 'paid', 'search', 'social', 'media', 'email',
                'marketing', 'affiliate', 'marketing', 'influencer', 'marketing', 'content', 'marketing',
                'inbound', 'marketing', 'outbound', 'marketing', 'growth', 'hacking', 'viral', 'marketing',
                'guerrilla', 'marketing', 'ambient', 'marketing', 'experiential', 'marketing', 'event',
                'marketing', 'trade', 'shows', 'conferences', 'exhibitions', 'sponsorships', 'partnerships',
                'alliances', 'joint', 'ventures', 'mergers', 'acquisitions', 'ipo', 'initial', 'public',
                'offering', 'venture', 'capital', 'private', 'equity', 'angel', 'investors', 'crowdfunding',
                'bootstrapping', 'funding', 'rounds', 'seed', 'series', 'a', 'series', 'b', 'series', 'c',
                'unicorn', 'startup', 'scaleup', 'growth', 'stage', 'mature', 'company', 'enterprise',
                'corporation', 'multinational', 'conglomerate', 'holding', 'company', 'subsidiary', 'division',
                'department', 'unit', 'team', 'group', 'committee', 'board', 'executive', 'management',
                'senior', 'management', 'middle', 'management', 'frontline', 'management', 'supervisor',
                'manager', 'director', 'vp', 'vice', 'president', 'svp', 'senior', 'vice', 'president',
                'evp', 'executive', 'vice', 'president', 'president', 'ceo', 'chief', 'executive', 'officer',
                'coo', 'chief', 'operating', 'officer', 'cfo', 'chief', 'financial', 'officer', 'cto',
                'chief', 'technology', 'officer', 'cmo', 'chief', 'marketing', 'officer', 'chro', 'chief',
                'human', 'resources', 'officer', 'cpo', 'chief', 'product', 'officer', 'cso', 'chief',
                'security', 'officer', 'cdo', 'chief', 'data', 'officer', 'cai', 'chief', 'ai', 'officer',
                'cgo', 'chief', 'growth', 'officer', 'cvo', 'chief', 'vision', 'officer', 'cbo', 'chief',
                'brand', 'officer', 'cco', 'chief', 'compliance', 'officer', 'cro', 'chief', 'risk', 'officer',
                'cfo', 'chief', 'fraud', 'officer', 'cfo', 'chief', 'future', 'officer', 'cfo', 'chief',
                'foresight', 'officer', 'cfo', 'chief', 'foresight', 'officer', 'cfo', 'chief', 'foresight',
                'officer', 'cfo', 'chief', 'foresight', 'officer', 'cfo', 'chief', 'foresight', 'officer'
            }
            
            logger.info("Enhanced preprocessor initialized successfully")
            
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text with enhanced processing, preserving newlines between logical lines
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        try:
            logger.info("Cleaning text with enhanced processing...")
            
            # Convert to lowercase
            text = text.lower()
            
            # Normalize line endings first
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            lines = [ln.strip() for ln in text.split('\n')]
            # Collapse superfluous spaces per line but keep line structure
            lines = [re.sub(r'\s+', ' ', ln) for ln in lines if ln is not None]
            text = '\n'.join(lines)
            
            # Remove special characters but keep alphanumeric, spaces, and common punctuation
            text = re.sub(r'[^\w\s\n\.\,\;\:\!\?\-\(\)\|]', ' ', text)
            
            # Remove multiple consecutive punctuation within lines
            text = re.sub(r'([\.\,\;\:\!\?]){2,}', r'\1', text)
            
            # Trim spaces around line breaks
            text = re.sub(r'\s*\n\s*', '\n', text)
            
            # Final trim
            text = text.strip()
            
            logger.info(f"Text cleaned successfully ({len(text)} characters)")
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            raise
    
    def extract_locations(self, text: str) -> List[str]:
        """
        Extract locations from text with focus on Indian cities
        
        Args:
            text (str): Text to extract locations from
            
        Returns:
            List[str]: List of locations found
        """
        try:
            logger.info("Extracting locations...")
            
            locations = set()
            
            # Extract using spaCy NER
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC']:  # Geopolitical entities and locations
                    locations.add(ent.text.strip())
            
            # Extract Indian cities using pattern matching
            words = text.lower().split()
            for i, word in enumerate(words):
                # Check for city names
                if word in self.indian_cities:
                    locations.add(word.title())
                
                # Check for city, state patterns
                if i < len(words) - 1:
                    city_state = f"{word} {words[i+1]}"
                    if city_state in self.indian_cities:
                        locations.add(city_state.title())
            
            # Extract location patterns
            location_patterns = [
                r'\b(bangalore|bengaluru|mumbai|delhi|hyderabad|chennai|pune|kolkata|ahmedabad|gurgaon|noida)\b',
                r'\b(india|usa|united states|america|canada|uk|united kingdom)\b',
                r'\b([a-z]+,\s*(india|usa|america|canada|uk))\b'
            ]
            
            for pattern in location_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        locations.add(match[0].title())
                    else:
                        locations.add(match.title())
            
            location_list = list(locations)
            location_list.sort()
            
            logger.info(f"Extracted {len(location_list)} locations: {location_list}")
            return location_list
            
        except Exception as e:
            logger.error(f"Error extracting locations: {str(e)}")
            raise
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills with improved filtering to avoid location confusion
        
        Args:
            text (str): Text to extract skills from
            
        Returns:
            List[str]: List of skills
        """
        try:
            logger.info("Extracting skills with enhanced filtering...")
            
            # Technical skills patterns (more comprehensive)
            skill_patterns = [
                # Programming Languages
                r'\b(python|java|javascript|typescript|react|angular|vue|node\.?js|express|php|ruby|go|rust|swift|kotlin|c\+\+|c#|\.net)\b',
                r'\b(html|css|bootstrap|sass|less|webpack|babel|jquery|ajax)\b',
                
                # Databases
                r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch|oracle|sqlite|dynamodb|cassandra)\b',
                
                # Cloud & DevOps
                r'\b(aws|azure|gcp|docker|kubernetes|jenkins|git|github|gitlab|terraform|ansible|linux|unix)\b',
                
                # Machine Learning & AI
                r'\b(machine learning|ml|deep learning|ai|artificial intelligence|nlp|natural language processing|computer vision|cv)\b',
                r'\b(tensorflow|pytorch|scikit-learn|pandas|numpy|matplotlib|seaborn|opencv|keras)\b',
                
                # Data Science
                r'\b(data science|data analysis|statistics|analytics|business intelligence|bi|etl|data visualization)\b',
                r'\b(tableau|power bi|excel|r language|spark|hadoop|kafka|airflow)\b',
                
                # Web Development
                r'\b(rest api|graphql|microservices|agile|scrum|devops|ci/cd|api development|web development)\b',
                r'\b(spring boot|django|flask|express|laravel|rails|asp\.net|jsp|servlets)\b',
                
                # Mobile Development
                r'\b(react native|flutter|android|ios|xamarin|cordova|ionic|swift|kotlin|objective-c)\b',
                
                # Testing & QA
                r'\b(selenium|jest|junit|testng|cypress|postman|soapui|manual testing|automated testing|qa)\b',
                
                # Design & UI/UX
                r'\b(ui design|ux design|figma|sketch|adobe|photoshop|illustrator|prototyping|wireframing)\b',
                
                # Cybersecurity
                r'\b(cybersecurity|network security|penetration testing|ethical hacking|security|encryption|cryptography)\b',
                
                # Blockchain
                r'\b(blockchain|solidity|web3|smart contracts|cryptocurrency|ethereum|bitcoin|defi|nft)\b',
                
                # Other Technologies
                r'\b(blockchain|iot|internet of things|embedded systems|arduino|raspberry pi|microcontrollers)\b',
                r'\b(ar|vr|augmented reality|virtual reality|unity|unreal engine|3d modeling|animation)\b'
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
                    token_lower = token.text.lower()
                    
                    # Check if it's likely a technical skill and not a location/non-skill word
                    if (token_lower not in self.non_skill_words and 
                        token_lower not in self.indian_cities and
                        any(keyword in token_lower for keyword in 
                            ['api', 'sql', 'js', 'ml', 'ai', 'dev', 'ops', 'web', 'app', 'data', 'tech', 'soft', 'hard'])):
                        skills.add(token.text)
            
            # Clean and normalize skills
            cleaned_skills = []
            for skill in skills:
                skill = skill.strip().lower()
                if (len(skill) > 1 and 
                    skill not in self.non_skill_words and 
                    skill not in self.indian_cities and
                    skill not in ['api', 'js', 'ml', 'ai']):
                    cleaned_skills.append(skill)
            
            # Remove duplicates and sort
            cleaned_skills = list(set(cleaned_skills))
            cleaned_skills.sort()
            
            logger.info(f"Extracted {len(cleaned_skills)} skills: {cleaned_skills[:10]}...")
            return cleaned_skills
            
        except Exception as e:
            logger.error(f"Error extracting skills: {str(e)}")
            raise
    
    def extract_certifications(self, text: str) -> List[str]:
        """
        Extract certifications from resume text
        
        Args:
            text (str): Text to extract certifications from
            
        Returns:
            List[str]: List of certifications
        """
        try:
            logger.info("Extracting certifications...")
            
            certifications = set()
            
            # Certification patterns
            cert_patterns = [
                r'\b(aws certified|azure certified|gcp certified|google certified|microsoft certified)\b',
                r'\b(pmp|scrum master|agile certified|itil|cisco ccna|ccnp|comptia|oracle certified)\b',
                r'\b(certified|certification|certificate|diploma|license|licensed)\b',
                r'\b(professional|specialist|expert|master|associate|foundation)\b'
            ]
            
            # Extract using patterns
            for pattern in cert_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                certifications.update(matches)
            
            # Extract using spaCy NER
            doc = self.nlp(text)
            for token in doc:
                if 'cert' in token.text.lower() or 'diploma' in token.text.lower():
                    certifications.add(token.text)
            
            cert_list = list(certifications)
            cert_list.sort()
            
            logger.info(f"Extracted {len(cert_list)} certifications")
            return cert_list
            
        except Exception as e:
            logger.error(f"Error extracting certifications: {str(e)}")
            raise
    
    def extract_experience_details(self, text: str) -> Dict[str, any]:
        """
        Extract detailed experience information
        
        Args:
            text (str): Text to extract experience from
            
        Returns:
            Dict[str, any]: Experience details
        """
        try:
            logger.info("Extracting experience details...")
            
            experience_info = {
                'companies': [],
                'positions': [],
                'duration': [],
                'technologies': []
            }
            
            # Company patterns
            company_patterns = [
                r'\b(worked at|worked for|employed at|employed by|intern at|interned at)\s+([a-zA-Z\s&.,]+)',
                r'\b(company|corporation|inc|llc|ltd|technologies|solutions|systems|services)\b'
            ]
            
            # Position patterns
            position_patterns = [
                r'\b(software engineer|developer|programmer|analyst|consultant|manager|lead|senior|junior|intern)\b',
                r'\b(frontend|backend|full stack|data scientist|ml engineer|devops|qa|tester|designer)\b'
            ]
            
            # Duration patterns
            duration_patterns = [
                r'\b(\d+)\s*(years?|months?|yrs?|mos?)\b',
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}'
            ]
            
            # Extract companies
            for pattern in company_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        experience_info['companies'].append(match[1].strip())
                    else:
                        experience_info['companies'].append(match.strip())
            
            # Extract positions
            for pattern in position_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                experience_info['positions'].extend(matches)
            
            # Extract duration
            for pattern in duration_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                experience_info['duration'].extend(matches)
            
            # Remove duplicates
            for key in experience_info:
                experience_info[key] = list(set(experience_info[key]))
            
            logger.info(f"Extracted experience: {len(experience_info['companies'])} companies, {len(experience_info['positions'])} positions")
            return experience_info
            
        except Exception as e:
            logger.error(f"Error extracting experience details: {str(e)}")
            raise
    
    def extract_projects(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract project information from resume with better precision
        
        Args:
            text (str): Text to extract projects from
            
        Returns:
            List[Dict[str, Any]]: List of project dictionaries
        """
        try:
            logger.info("Extracting projects...")
            
            projects = []
            
            # More precise project patterns - look for project titles/names only
            project_patterns = [
                # Project title patterns
                r'\b(project|project name|title)\s*:?\s*([A-Z][a-zA-Z0-9\s\-_&]+)',
                r'\b(developed|built|created|designed|implemented)\s+([A-Z][a-zA-Z0-9\s\-_&]+)',
                # Look for capitalized project names (likely titles)
                r'\b([A-Z][a-zA-Z0-9\s\-_&]{3,})\s+(?:project|application|system|platform|tool)',
                # Specific project types
                r'\b([A-Z][a-zA-Z0-9\s\-_&]+)\s+(?:web app|mobile app|desktop app|ML model|dashboard|API)'
            ]
            
            # Extract project titles only
            for pattern in project_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        project_name = match[1].strip()
                    else:
                        project_name = match.strip()
                    
                    # Filter out common non-project words
                    if (len(project_name) > 3 and 
                        not any(word in project_name.lower() for word in ['the', 'and', 'or', 'but', 'with', 'using', 'technologies', 'tech stack'])):
                        projects.append({
                            'name': project_name,
                            'description': '',
                            'technologies': []
                        })
            
            # Remove duplicates based on name
            unique_projects = []
            seen_names = set()
            for project in projects:
                if project['name'].lower() not in seen_names:
                    unique_projects.append(project)
                    seen_names.add(project['name'].lower())
            
            logger.info(f"Extracted {len(unique_projects)} projects")
            return unique_projects
            
        except Exception as e:
            logger.error(f"Error extracting projects: {str(e)}")
            raise
    
    def extract_achievements(self, text: str) -> List[str]:
        """
        Extract achievements from resume
        
        Args:
            text (str): Text to extract achievements from
            
        Returns:
            List[str]: List of achievements
        """
        try:
            logger.info("Extracting achievements...")
            
            achievements = set()
            
            # Achievement patterns
            achievement_patterns = [
                r'\b(achieved|accomplished|completed|won|awarded|recognized|honored|selected)\b',
                r'\b(first place|second place|third place|winner|runner-up|finalist|top|best)\b',
                r'\b(published|presented|speaker|mentor|volunteer|leadership|team lead|project lead)\b',
                r'\b(gpa|grade point average|percentage|rank|position|score|rating)\b'
            ]
            
            # Extract using patterns
            for pattern in achievement_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                achievements.update(matches)
            
            achievement_list = list(achievements)
            achievement_list.sort()
            
            logger.info(f"Extracted {len(achievement_list)} achievements")
            return achievement_list
            
        except Exception as e:
            logger.error(f"Error extracting achievements: {str(e)}")
            raise
    
    def extract_field_of_interest(self, text: str) -> List[str]:
        """
        Extract field of interest from resume
        
        Args:
            text (str): Text to extract field of interest from
            
        Returns:
            List[str]: List of fields of interest
        """
        try:
            logger.info("Extracting field of interest...")
            
            fields = set()
            
            # Field patterns
            field_patterns = [
                r'\b(interested in|passionate about|love|enjoy|focus on|specialize in|expertise in)\s+([a-zA-Z\s]+)',
                r'\b(software engineering|data science|machine learning|web development|mobile development|cybersecurity)\b',
                r'\b(artificial intelligence|blockchain|cloud computing|devops|ui/ux|product management|business analysis)\b',
                r'\b(fintech|healthtech|edtech|agritech|greentech|iot|ar/vr|game development)\b'
            ]
            
            # Extract using patterns
            for pattern in field_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        fields.add(match[1].strip())
                    else:
                        fields.add(match.strip())
            
            field_list = list(fields)
            field_list.sort()
            
            logger.info(f"Extracted {len(field_list)} fields of interest")
            return field_list
            
        except Exception as e:
            logger.error(f"Error extracting field of interest: {str(e)}")
            raise
    
    def segment_sections(self, text_lines: List[str]) -> Dict[str, List[str]]:
        """Segment lines into sections based on heading keywords (case-insensitive).
        Returns dict mapping canonical section names to list of lines.
        """
        try:
            logger.info("Segmenting resume into sections (rule-based)...")
            # Prepare helper to match headings
            def normalize(s: str) -> str:
                return re.sub(r'[^a-z]+', ' ', s.lower()).strip()
            # Build reverse map: keyword -> canonical section
            keyword_to_section = {}
            for canon, aliases in self.section_aliases.items():
                for a in aliases:
                    keyword_to_section[a] = canon
            sections: Dict[str, List[str]] = {k: [] for k in self.section_aliases.keys()}
            current: str = None
            for raw in text_lines:
                ln = raw.strip()
                if not ln:
                    continue
                norm = normalize(ln)
                # Check explicit heading match (word equals or ends with ':'), also ALLCAPS heuristic
                is_allcaps = ln.isupper() and len(ln) <= 60 and any(c.isalpha() for c in ln)
                matched = None
                if is_allcaps or ln.endswith(':') or len(ln.split()) <= 4:
                    for kw, target in keyword_to_section.items():
                        if norm == kw or norm.startswith(kw) or kw in norm:
                            matched = target
                            break
                if matched:
                    current = matched
                    logger.info(f"Detected section heading: {ln} -> {current}")
                    continue
                # Append to current section if known, else ignore (we won't mix across)
                if current in sections:
                    sections[current].append(ln)
            return sections
        except Exception as e:
            logger.warning(f"Section segmentation failed: {e}")
            return {k: [] for k in self.section_aliases.keys()}

    def _extract_projects_from_section(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract projects titles from Projects section; ignore descriptive bullet points."""
        projects = []
        try:
            if not lines:
                return projects
            # Potential title lines: not starting with bullet, 2-10 words, avoid verbs
            verbs = {'built','developed','designed','implemented','created','optimized','tested','managed','automated','engineered'}
            for ln in lines:
                lns = ln.strip().strip('-•–—').strip()
                if not lns:
                    continue
                tokens = re.findall(r'[A-Za-z][A-Za-z0-9\-]+', lns)
                if not (2 <= len(tokens) <= 10):
                    continue
                low = lns.lower()
                if any(v in low for v in verbs):
                    continue
                # Prefer lines that look like titles (Title Case or contain system/app/application/platform/dashboard)
                if lns.istitle() or re.search(r'\b(system|application|app|platform|dashboard|portal|engine)\b', low):
                    name = lns
                    if name.lower() not in {p['name'].lower() for p in projects}:
                        projects.append({'name': name, 'description': '', 'technologies': []})
            return projects
        except Exception as e:
            logger.warning(f"Project extraction failed: {e}")
            return projects

    def _extract_education_from_section(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract education entries from Education section."""
        entries: List[Dict[str, Any]] = []
        try:
            degree_pat = re.compile(r"(Bachelor|Master|B\.?Sc|M\.?Sc|B\.?E|M\.?E|B\.?Tech|M\.?Tech|BCA|MCA|MBA|BBA|BCom|MCom|Diploma)", re.I)
            date_pat = re.compile(r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}\b)|\b\d{4}\b", re.I)
            for i, ln in enumerate(lines):
                has_degree = degree_pat.search(ln)
                has_inst = re.search(r"\b(College|University|School|Institute)\b", ln, re.I)
                if has_degree or has_inst:
                    entry = {'degree': '', 'institution': '', 'dates': '', 'grade': ''}
                    if has_degree:
                        entry['degree'] = has_degree.group(0)
                    # Institution may be same line or next non-empty line
                    inst_line = ln if has_inst else next((l for l in lines[i+1:i+3] if re.search(r"\b(College|University|School|Institute)\b", l, re.I)), '')
                    if inst_line:
                        entry['institution'] = inst_line.strip()
                    # Dates
                    dates = date_pat.findall(ln) or date_pat.findall(' '.join(lines[i:i+2]))
                    if dates:
                        flat = [d if isinstance(d, str) else d[0] for d in dates]
                        entry['dates'] = ' - '.join(sorted(set([d for d in flat if d])))
                    if entry['degree'] or entry['institution']:
                        entries.append(entry)
            return entries
        except Exception as e:
            logger.warning(f"Education extraction failed: {e}")
            return entries

    def _extract_certifications_from_section(self, lines: List[str]) -> List[Dict[str, Any]]:
        certs: List[Dict[str, Any]] = []
        try:
            if not lines:
                return certs
            issuers = ['guvi','hackerrank','coursera','udemy','edx','google','microsoft','ibm','oracle']
            for ln in lines:
                if not ln.strip():
                    continue
                m_issuer = re.search(r'\b(' + '|'.join(issuers) + r')\b', ln, re.I)
                issuer = m_issuer.group(1).title() if m_issuer else ''
                name = ln.strip().strip('-•–—').strip()
                if name:
                    certs.append({'name': name, 'issuer': issuer, 'date': ''})
            return certs
        except Exception as e:
            logger.warning(f"Certification extraction failed: {e}")
            return certs

    def _extract_skills_from_section(self, lines: List[str]) -> List[str]:
        try:
            txt = '\n'.join(lines)
            return self.extract_skills(txt)
        except Exception as e:
            logger.warning(f"Skills section extraction failed: {e}")
            return []

    def _extract_achievements_from_section(self, lines: List[str]) -> List[str]:
        """Extract achievement bullet lines from Achievements section using simple heuristics, splitting compound lines into multiple items."""
        items: List[str] = []
        try:
            import re
            pat = re.compile(r"((?:Winner|Finalist|\d{1,2}(?:st|nd|rd|th)\s+Runner(?:-?Up)?)\s+[\-–—]\s+.+?\(\d{4}\))")
            def split_items(ln: str) -> List[str]:
                s = (ln or '').strip()
                if not s:
                    return []
                s = s.lstrip('•-* ').strip()
                found = [m.group(1).strip() for m in pat.finditer(s)]
                if not found:
                    # fallback if contains a year
                    if re.search(r'\b(20\d{2}|19\d{2})\b', s):
                        return [s]
                    return []
                # de-dup preserve order
                seen = set()
                out = []
                for f in found:
                    low = f.lower()
                    if low not in seen:
                        out.append(f)
                        seen.add(low)
                return out
            for ln in lines:
                items.extend(split_items(ln))
            # de-dup across lines while preserving order
            seen = set()
            uniq = []
            for s in items:
                low = s.lower()
                if low not in seen:
                    uniq.append(s)
                    seen.add(low)
            return uniq
        except Exception as e:
            logger.warning(f"Achievements section extraction failed: {e}")
            return items

    def preprocess_resume(self, text: str) -> Dict[str, any]:
        """
        Complete preprocessing pipeline for resume text with enhanced extraction
        
        Args:
            text (str): Raw resume text
            
        Returns:
            Dict[str, any]: Dictionary containing all extracted information
        """
        try:
            logger.info("Starting enhanced resume preprocessing pipeline...")
            
            # Clean text (preserve line structure) and segment into sections
            cleaned_text = self.clean_text(text)
            lines = cleaned_text.split('\n')
            sections = self.segment_sections(lines)

            # Section-aware extraction first
            skills_sec = self._extract_skills_from_section(sections.get('skills', []))
            projects_sec = self._extract_projects_from_section(sections.get('projects', []))
            certs_sec = self._extract_certifications_from_section(sections.get('certifications', []))
            education_sec = self._extract_education_from_section(sections.get('education', []))
            achievements_sec = self._extract_achievements_from_section(sections.get('achievements', []))

            # Global fallback extraction to augment missing items
            locations = self.extract_locations(cleaned_text)
            skills_global = self.extract_skills(cleaned_text)
            certifications_global = self.extract_certifications(cleaned_text)
            experience_details = self.extract_experience_details(cleaned_text)
            projects_global = self.extract_projects(cleaned_text)
            achievements_global = self.extract_achievements(cleaned_text)
            field_of_interest = self.extract_field_of_interest(cleaned_text)

            # Merge section-aware with global (section results take precedence)
            skills = sorted(list(set((skills_sec or []) + (skills_global or []))))
            projects = projects_sec if projects_sec else projects_global
            certifications = certs_sec if certs_sec else certifications_global
            achievements = achievements_sec if achievements_sec else achievements_global

            # Extract entities using spaCy
            entities = self.extract_entities(cleaned_text)
            
            # Tokenize and lemmatize
            tokens = self.tokenize_and_lemmatize(cleaned_text)
            
            # Create result dictionary
            result = {
                'cleaned_text': cleaned_text,
                'tokens': tokens,
                'entities': entities,
                'skills': skills,
                'locations': locations,
                'certifications': certifications,
                'education': education_sec,  # newly added, section-aware
                'experience_details': experience_details,
                'projects': projects,
                'achievements': achievements,
                'field_of_interest': field_of_interest,
                'text_length': len(cleaned_text),
                'token_count': len(tokens)
            }
            
            logger.info("Enhanced resume preprocessing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced resume preprocessing: {str(e)}")
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
                'GPE': [],
                'DATE': [],
                'MONEY': [],
                'PERCENT': [],
                'TIME': [],
                'EVENT': [],
                'FAC': [],
                'LANGUAGE': [],
                'LAW': [],
                'LOC': [],
                'NORP': [],
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


# Backward compatibility
class TextPreprocessor(EnhancedTextPreprocessor):
    """Backward compatibility wrapper"""
    pass


# Convenience functions for easy usage
def preprocess_resume_text(text: str) -> Dict[str, any]:
    """
    Convenience function to preprocess resume text
    
    Args:
        text (str): Raw resume text
        
    Returns:
        Dict[str, any]: Preprocessed resume data
    """
    preprocessor = EnhancedTextPreprocessor()
    return preprocessor.preprocess_resume(text)


def preprocess_job_description_text(text: str) -> Dict[str, any]:
    """
    Convenience function to preprocess job description text
    
    Args:
        text (str): Raw job description text
        
    Returns:
        Dict[str, any]: Preprocessed job description data
    """
    preprocessor = EnhancedTextPreprocessor()
    return preprocessor.preprocess_job_description(text)


if __name__ == "__main__":
    # Test the enhanced text preprocessor
    preprocessor = EnhancedTextPreprocessor()
    
    # Example usage
    sample_text = """
    John Smith
    Software Engineer
    Experience: 3 years in Python, JavaScript, and web development
    Education: Bachelor's in Computer Science from MIT
    Skills: Machine Learning, Data Science, React, Node.js
    Location: Bangalore, India
    Certifications: AWS Certified Solutions Architect
    Projects: E-commerce website using React and Node.js
    Achievements: Won first place in hackathon
    """
    
    print("ResuMatch AI - Enhanced Text Preprocessor")
    print("Testing enhanced preprocessing pipeline...")
    
    result = preprocessor.preprocess_resume(sample_text)
    print(f"Extracted skills: {result['skills']}")
    print(f"Extracted locations: {result['locations']}")
    print(f"Extracted certifications: {result['certifications']}")
    print(f"Extracted projects: {result['projects']}")
    print(f"Extracted achievements: {result['achievements']}")
    print(f"Field of interest: {result['field_of_interest']}")
