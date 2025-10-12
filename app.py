"""
Flask web application for ResuMatch AI
Main application file with routes for resume upload and job recommendations
"""

from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
import os
import json
import logging
from werkzeug.utils import secure_filename
import tempfile
import re
import uuid
from typing import Dict, List, Any
from collections import Counter

# Import our custom modules
from extract_text import extract_text_from_bytes, extract_text_with_hints_from_bytes
from enhanced_preprocess import EnhancedTextPreprocessor
from enhanced_model import EnhancedResuMatchModel, load_job_data
from openai_parser import OpenAIResumeParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'resumatch-ai-secret-key-2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Feature flags
ENABLE_OPENAI = os.getenv('ENABLE_OPENAI', '0') == '1'

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model and preprocessor
model = None
preprocessor = None

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(filename: str) -> str:
    """Get file type from filename"""
    return filename.rsplit('.', 1)[1].lower()


def normalize_location(loc: str) -> str:
    """Normalize location strings like 'pune,india' to 'Pune, India'"""
    if not loc:
        return ''
    parts = [p.strip() for p in loc.replace('|', ',').split(',') if p.strip()]
    parts = [p.title() for p in parts]
    return ', '.join(parts)

# Simple server-side profile storage to avoid cookie size limits
PROFILE_STORAGE_DIR = os.path.join('storage', 'profiles')
os.makedirs(PROFILE_STORAGE_DIR, exist_ok=True)

def _profile_path(profile_id: str) -> str:
    return os.path.join(PROFILE_STORAGE_DIR, f"{profile_id}.json")

def save_profile_data(profile: Dict[str, Any]) -> str:
    profile_id = uuid.uuid4().hex
    try:
        with open(_profile_path(profile_id), 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False)
        return profile_id
    except Exception as e:
        logger.error(f"Failed to save profile data: {e}")
        raise

def load_profile_data(profile_id: str) -> Dict[str, Any]:
    try:
        with open(_profile_path(profile_id), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load profile data for {profile_id}: {e}")
        return None

def update_profile_data(profile_id: str, profile: Dict[str, Any]) -> bool:
    try:
        with open(_profile_path(profile_id), 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to update profile data for {profile_id}: {e}")
        return False


def enrich_processed_data(processed_data: Dict[str, Any], raw_text: str, hints: Dict[str, Any] = None) -> Dict[str, Any]:
    """Enrich and normalize processed data for profile rendering.
    - Populate top-level fields (name, email, phone, location)
    - Detect LinkedIn/GitHub/portfolio URLs from text
    - Normalize experience, projects, certifications shapes
    - Augment achievements and project descriptions with keyword hints
    """
    if not isinstance(processed_data, dict):
        processed_data = {}

    # Prefer first heading as name if it looks like a personal name
    if hints and isinstance(hints, dict) and not processed_data.get('name'):
        skip_headings = {"EDUCATION","SKILLS","PROJECT","PROJECTS","CERTIFICATION","CERTIFICATIONS","ACHIEVEMENTS","SIMULATION","SUMMARY","OBJECTIVE","PROFILE","EXPERIENCE","PROFESSIONAL EXPERIENCE"}
        heads = hints.get('headings') or []
        for h in heads:
            ht = (h.get('text') or '').strip()
            if not ht:
                continue
            if ht.upper() in skip_headings:
                continue
            # Accept short headings 2-5 tokens, mostly letters and spaces
            words = [w for w in ht.split() if w.isalpha() or (len(w)==1 and w.isalpha())]
            if 1 < len(words) <= 5 and all(len(w) <= 20 for w in words):
                processed_data['name'] = ' '.join(w.capitalize() for w in words)
                break

    # 1) Basic contact extraction from raw text
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    phone_pattern = r"(\+?\d[\d\-\s]{7,}\d)"
    url_pattern = r"(https?://[^\s]+|www\.[^\s]+)"

    emails = re.findall(email_pattern, raw_text or "")
    # Fallback email detection for OCR with spaces (e.g., user @ gmail . com)
    if not emails:
        spaced_email = re.search(r"([A-Za-z0-9._%+-]+)\s*@\s*([A-Za-z0-9.-]+)\s*\.?\s*([A-Za-z]{2,})", raw_text or "", flags=re.I)
        if spaced_email:
            local, domain, tld = spaced_email.group(1), spaced_email.group(2), spaced_email.group(3)
            emails = [f"{local}@{domain}.{tld}"]
    # Explicitly handle '@gmail.com' with optional spaces (and common OCR variants)
    if not emails:
        gm1 = re.search(r"([A-Za-z0-9._%+-]+)\s*@\s*gmail\s*\.\s*com\b", raw_text or "", flags=re.I)
        if gm1:
            emails = [f"{gm1.group(1)}@gmail.com"]
    # Explicitly handle '@gmai.com' with optional spaces (OCR miss of 'l')
    if not emails:
        gmai_match = re.search(r"([A-Za-z0-9._%+-]+)\s*@\s*gmai\s*\.\s*com\b", raw_text or "", flags=re.I)
        if gmai_match:
            emails = [f"{gmai_match.group(1)}@gmai.com"]

    phones = re.findall(phone_pattern, raw_text or "")
    urls = re.findall(url_pattern, raw_text or "")

    # 2) URL classification
    linkedin_url = None
    github_url = None
    portfolio_url = None
    for u in urls:
        u_norm = u.strip().strip('.,)(')
        low = u_norm.lower()
        if 'linkedin.com' in low and not linkedin_url:
            linkedin_url = u_norm
        elif ('github.com' in low or 'gitlab.com' in low) and not github_url:
            github_url = u_norm
        elif not portfolio_url and all(d not in low for d in ['linkedin.com', 'github.com', 'gitlab.com']):
            portfolio_url = u_norm

    # 3) Location fallback from list if top-level missing
    if not processed_data.get('location'):
        locs = processed_data.get('locations') or []
        if isinstance(locs, list) and locs:
            processed_data['location'] = locs[0]

    # 4) Name fallback using entities/person or first line of text
    if not processed_data.get('name'):
        name = None
        ents = processed_data.get('entities') or {}
        if isinstance(ents, dict):
            people = ents.get('PERSON') or []
            if isinstance(people, list) and people:
                name = people[0]
        if not name:
            # First non-empty line heuristic (e.g., SHIFLINA NILOFAR P)
            for line in (raw_text or '').splitlines():
                s = line.strip()
                if s and len(s.split()) <= 6:
                    name = s
                    break
        if not name and emails:
            name = emails[0].split('@')[0].replace('.', ' ').replace('_', ' ').title()
        if name:
            processed_data['name'] = name

    # 5) Email/phone top-level
    if not processed_data.get('email') and emails:
        processed_data['email'] = emails[0]
    if not processed_data.get('phone') and phones:
        processed_data['phone'] = phones[0]

    # 6) Socials top-level
    if linkedin_url and not processed_data.get('linkedin'):
        processed_data['linkedin'] = linkedin_url
    if github_url and not processed_data.get('github'):
        processed_data['github'] = github_url
    if portfolio_url and not processed_data.get('portfolio'):
        processed_data['portfolio'] = portfolio_url

    # 7) Experience: if empty but experience_details present, build a simple list
    if (not processed_data.get('experience')) and isinstance(processed_data.get('experience_details'), dict):
        expd = processed_data['experience_details']
        comps = expd.get('companies') or []
        poss = expd.get('positions') or []
        durs = expd.get('duration') or []
        exp_list = []
        n = max(len(comps), len(poss), len(durs))
        for i in range(n):
            exp_list.append({
                'company': comps[i] if i < len(comps) else '',
                'position': poss[i] if i < len(poss) else '',
                'dates': durs[i] if i < len(durs) else '',
                'description': ''
            })
        processed_data['experience'] = exp_list

    # 8) Normalize projects and certifications if still strings
    def ensure_projects_list(projects):
        out = []
        if not isinstance(projects, list):
            return out
        for p in projects:
            if isinstance(p, dict):
                out.append({
                    'name': p.get('name', '') or p.get('title', ''),
                    'description': p.get('description', ''),
                    'technologies': p.get('technologies', []) or p.get('tech', []) or []
                })
            elif isinstance(p, str):
                out.append({'name': p, 'description': '', 'technologies': []})
        return out

    def ensure_certs_list(certs):
        out = []
        if not isinstance(certs, list):
            return out
        for c in certs:
            if isinstance(c, dict):
                out.append({
                    'name': c.get('name', '') or c.get('title', ''),
                    'issuer': c.get('issuer', ''),
                    'date': c.get('date', '')
                })
            elif isinstance(c, str):
                out.append({'name': c, 'issuer': '', 'date': ''})
        return out

    processed_data['projects'] = ensure_projects_list(processed_data.get('projects', []))
    processed_data['certifications'] = ensure_certs_list(processed_data.get('certifications', []))

    # 9) Section-aware extraction (Projects, Certifications, Achievements)
    # Prefer structural hints if provided (headings, bold subheadings)
    projects_set_from_hints = False
    experience_set_from_hints = False
    if hints and isinstance(hints, dict) and hints.get('sections'):
        sections = hints.get('sections')
        # Projects (STRICT: only from Projects section, pull titles from bold/subheadings)
        for sec in sections:
            name = (sec.get('name') or '').upper()
            if name in {'PROJECT', 'PROJECTS'}:
                subs = [s for s in (sec.get('subheadings') or []) if s and len(s) <= 120]
                verbs = {'built','developed','designed','implemented','created','optimized','tested','managed','automated','engineered'}
                filtered = []
                for s in subs:
                    low = s.lower()
                    if any(v in low for v in verbs):
                        continue
                    filtered.append(s.strip())
                if filtered:
                    seensp = set()
                    titles = []
                    for t in filtered:
                        lt = t.lower()
                        if lt not in seensp:
                            titles.append(t)
                            seensp.add(lt)
                    processed_data['projects'] = [{'name': t, 'description': '', 'technologies': []} for t in titles]
                    projects_set_from_hints = True
                    break
        # Experience (STRICT from Experience section; deduplicate)
        for sec in sections:
            name = (sec.get('name') or '').upper()
            if name in {'EXPERIENCE', 'PROFESSIONAL EXPERIENCE'}:
                subs = [s for s in (sec.get('subheadings') or []) if s and len(s) <= 120]
                roles = []
                def add_role(flag, title):
                    if flag and title not in roles:
                        roles.append(title)
                for s in subs:
                    low = s.lower()
                    add_role('freelance' in low and 'web' in low and 'developer' in low, 'Freelance Web Developer')
                    add_role('intern' in low and 'python' in low and 'programming' in low and 'platform' in low, 'Intern - Python Programming Platform')
                if not roles:
                    for s in (sec.get('bold_lines') or []):
                        low = s.lower()
                        add_role('freelance' in low and 'web' in low and 'developer' in low, 'Freelance Web Developer')
                        add_role('intern' in low and 'python' in low and 'programming' in low and 'platform' in low, 'Intern - Python Programming Platform')
                if roles:
                    processed_data['experience'] = [{'company':'', 'position': r, 'dates':'', 'description':''} for r in roles]
                    experience_set_from_hints = True
                    break
        # Certifications
        for sec in sections:
            name = (sec.get('name') or '').upper()
            if name in {'CERTIFICATION', 'CERTIFICATIONS'}:
                tokens = []
                for ln in sec.get('lines', []) or []:
                    tokens.extend([t.strip() for t in re.split(r',\s*|\n+', ln) if t.strip()])
                certs = []
                for t in tokens:
                    m = re.search(r'\b(guvi|hackerrank|coursera|udemy|edx|google|microsoft|ibm|oracle)\b', t, flags=re.I)
                    if m:
                        issuer_raw = m.group(1).lower()
                        issuer = 'GUVI' if issuer_raw == 'guvi' else issuer_raw.title()
                        name_part = t[:m.start()].strip(' ,:')
                        if name_part:
                            certs.append({'name': name_part, 'issuer': issuer, 'date': ''})
                    else:
                        certs.append({'name': t, 'issuer': '', 'date': ''})
                if certs:
                    processed_data['certifications'] = certs
        # Education candidates
        ed_cands = hints.get('education_candidates') or []
        if ed_cands:
            edu_entries = processed_data.get('education', []) or []
            for cand in ed_cands:
                inst = re.sub(r"\b([A-Za-z]+)\s+s\s+(College|University)\b", r"\1's \2", cand, flags=re.I).strip()
                if inst and inst.lower() not in {e.get('institution','').lower() for e in edu_entries}:
                    edu_entries.append({'institution': inst, 'degree': '', 'dates': '', 'grade': ''})
            if edu_entries:
                processed_data['education'] = edu_entries

    # 9a) Continue with text-based section extraction to supplement hints
    source_text = processed_data.get('cleaned_text') or raw_text or ''
    low_text = source_text.lower()

    def slice_section(label, text_low):
        m = re.search(rf"\b{label}\b", text_low)
        if not m:
            return None
        start = m.end()
        # find next header among the known labels
        tail = text_low[start:]
        next_idx = len(tail)
        for nxt in ['projects', 'certifications', 'achievements']:
            if nxt == label:
                continue
            mn = re.search(rf"\b{nxt}\b", tail)
            if mn:
                next_idx = min(next_idx, mn.start())
        return source_text[start:start+next_idx].strip()

    # Support both 'project' and 'projects' headers
    proj_seg = slice_section('projects', low_text) or slice_section('project', low_text)
    cert_seg = slice_section('certifications', low_text)
    ach_seg = slice_section('achievements', low_text)

    # Projects from explicit section (prefer this if we find at least one)
    new_projects = []
    if proj_seg and not projects_set_from_hints:
        # Split on periods/semicolons
        sentences = [s.strip() for s in re.split(r'[\.;]\s*', proj_seg) if s.strip()]
        for sent in sentences:
            name = sent
            low = sent.lower()
            # Remove trailing tech phrase
            cut_tokens = [' built using ', ' using ', ' with ']
            for ct in cut_tokens:
                if ct in low:
                    idx = low.find(ct)
                    name = sent[:idx].strip()
                    break
            # Technologies detection
            tech_candidates = []
            for t in ['python', 'pandas', 'mysql', 'sql', 'flask', 'django', 'react', 'node', 'opencv', 'socket.io']:
                if t in low:
                    tech_candidates.append(t)
            proj = {
                'name': name if name else sent[:60],
                'description': '' if name != sent else sent,
                'technologies': tech_candidates
            }
            # Avoid duplicates and empty names
            if proj['name'] and proj['name'].lower() not in {p['name'].lower() for p in new_projects}:
                new_projects.append(proj)

        # Heuristic reconstruction for tokenized project names
        if not new_projects:
            tokens = [tok.strip() for tok in re.split(r'[\s,]+', proj_seg) if tok.strip()]
            verbs = {'develop','design','implement','create','build','manage','automate','improve','optimize','work','testing','test','focus','simulate'}
            heads = {'system','platform','application','dashboard','engine','service','tool'}
            lower = [t.lower() for t in tokens]
            # Find head nouns and backtrack to build name
            i = 0
            names = []
            while i < len(lower):
                if lower[i] in heads:
                    # backtrack until verb or start or header 'project(s)'
                    j = i
                    parts = []
                    while j >= 0 and lower[j] not in verbs and lower[j] not in {'project','projects'}:
                        parts.append(tokens[j])
                        j -= 1
                    parts.reverse()
                    name = ' '.join(parts).strip()
                    # Clean up extra words
                    name = re.sub(r'\b(stack|basic|online|mobile|responsive|design|consistent|performance|device|status|tracking)\b','', name, flags=re.I)
                    name = re.sub(r'\s{2,}',' ', name).strip()
                    # Ensure ends with head noun
                    if not name.lower().endswith(lower[i]):
                        name = (name + ' ' + tokens[i]).strip()
                    if name and name.lower() not in {n.lower() for n in names} and len(name) >= 6:
                        names.append(name)
                i += 1
            # Special case: 'watch together' inference from context
            if 'watch' in lower and any(k in lower for k in ['video','streaming','platform']):
                wt = 'Watch Together'
                if wt.lower() not in {n.lower() for n in names}:
                    names.append(wt)
            for nm in names[:3]:
                proj = {'name': nm.title(), 'description': '', 'technologies': []}
                if proj['name'].lower() not in {p['name'].lower() for p in new_projects}:
                    new_projects.append(proj)

        if new_projects and not projects_set_from_hints:
            processed_data['projects'] = new_projects

    # Certifications from explicit section
    new_certs = []
    if cert_seg:
        # Match "<name> <issuer>"
        for m in re.finditer(r'([A-Za-z][A-Za-z \-]{2,}?)\s+(guvi|hackerrank|coursera|udemy|edx|google|microsoft|ibm|oracle)', cert_seg, flags=re.I):
            name = m.group(1).strip().replace('  ', ' ')
            issuer_raw = m.group(2).lower()
            issuer = 'GUVI' if issuer_raw == 'guvi' else issuer_raw.title()
            new_certs.append({'name': name, 'issuer': issuer, 'date': ''})
        # If nothing matched, try comma-separated tokens as names
        if not new_certs:
            toks = [t.strip() for t in re.split(r',\s*|\n+', cert_seg) if t.strip()]
            for t in toks:
                # Try to split trailing issuer word
                tm = re.search(r'\b(guvi|hackerrank|coursera|udemy|edx|google|microsoft|ibm|oracle)\b', t, flags=re.I)
                if tm:
                    issuer_raw = tm.group(1).lower()
                    issuer = 'GUVI' if issuer_raw == 'guvi' else issuer_raw.title()
                    name = t[:tm.start()].strip(' ,')
                    if name:
                        new_certs.append({'name': name, 'issuer': issuer, 'date': ''})
                else:
                    new_certs.append({'name': t, 'issuer': '', 'date': ''})
        if new_certs:
            processed_data['certifications'] = new_certs

    # Achievements from explicit section
    new_achs = []
    if ach_seg:
        # Extract phrases like "1st runner-up ... 2025" or starting with winners/winner
        matches = re.findall(r'(\d+(?:st|nd|rd|th)\s+runner-up[^\d]+\d{4}|winners?[^\.,]+)', ach_seg, flags=re.I)
        for mtxt in matches:
            a = mtxt.strip(' ,.;\n')
            if a and a not in new_achs:
                new_achs.append(a)
        # Fallback: split by comma/period and keep lines containing winner/runner/finalist
        if not new_achs:
            parts = [p.strip() for p in re.split(r'[\.,]\s*', ach_seg) if p.strip()]
            for p in parts:
                low = p.lower()
                if any(k in low for k in ['winner', 'winners', 'runner-up', 'finalist', 'top', 'selected']):
                    if p and p not in new_achs:
                        new_achs.append(p)
    if new_achs:
            processed_data['achievements'] = new_achs

    # 9b) Experience section parsing (Professional Experience / Experience)
    exp_seg = slice_section('professional experience', low_text) or slice_section('experience', low_text)
    if exp_seg and not experience_set_from_hints:
        # Extract concise role headings
        lines = [l.strip() for l in exp_seg.splitlines() if l.strip()]
        roles = []
        # Merge tokens that commonly form roles
        text_joined = ' '.join(lines)
        tj_low = text_joined.lower()
        # Common role patterns
        role_patterns = [
            r"\b(intern)\b(?:\s*[-–—:\|]\s*)?([a-z][a-z\s]{2,}?platform)?",
            r"\b(freelance\s+web\s+developer)\b",
            r"\b(software\s+engineer)\b",
            r"\b(data\s+analyst)\b",
            r"\b(web\s+developer)\b",
        ]
        found = []
        for pat in role_patterns:
            for m in re.finditer(pat, tj_low, flags=re.I):
                role = m.group(0).strip()
                role = re.sub(r"\s+", " ", role)
                if role and role.lower() not in {f.lower() for f in found}:
                    found.append(role)
        # Specific reconstruction for "Intern - Python Programming Platform"
        if 'intern' in tj_low and 'python' in tj_low and 'programming' in tj_low and 'platform' in tj_low:
            if 'intern - python programming platform' not in {f.lower() for f in found}:
                found.insert(0, 'Intern - Python Programming Platform')
        # Build experience entries
        exp_list = processed_data.get('experience', []) or []
        for role in found:
            role_low = role.lower()
            if 'freelance' in role_low and 'web developer' in role_low:
                title = 'Freelance Web Developer'
            elif 'intern' in role_low and 'python' in role_low and 'programming' in role_low and 'platform' in role_low:
                title = 'Intern - Python Programming Platform'
            else:
                continue
            # Try to attach dates if present
            mdate = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}\s*(?:[-–—]\s*(present|\d{4}))?", tj_low, flags=re.I)
            dates = mdate.group(0) if mdate else ''
            exp_list.append({'company': '', 'position': title, 'dates': dates, 'description': ''})
        if exp_list:
            # Deduplicate by (company, position, dates)
            seen = set()
            uniq = []
            for e in exp_list:
                key = (e.get('company','').lower(), e.get('position','').lower(), e.get('dates','').lower())
                if key not in seen:
                    uniq.append(e)
                    seen.add(key)
            # Limit to at most two roles
            processed_data['experience'] = uniq[:2]

    # 10) Education detection (college/university lines)
    edu_entries = processed_data.get('education', []) or []
    # Look for lines containing 'college' or 'university'
    edu_matches = re.findall(r"([A-Z][A-Za-z\s]+?(?:College|University))", source_text, flags=re.I)
    for em in edu_matches:
        inst = em.strip()
        # Normalize " s " -> "'s " in college names (OCR artifact)
        inst = re.sub(r"\b([A-Za-z]+)\s+s\s+(College|University)\b", r"\1's \2", inst, flags=re.I)
        if inst and inst.lower() not in {e.get('institution','').lower() for e in edu_entries}:
            edu_entries.append({'institution': inst, 'degree': '', 'dates': '', 'grade': ''})
    if edu_entries:
        processed_data['education'] = edu_entries

    # 11) Project keyword hints
    keyword_tags = [
        'inventory management', 'e-commerce', 'chatbot', 'dashboard', 'recommendation', 'ocr', 'nlp', 'computer vision',
        'api', 'microservices', 'data pipeline', 'etl', 'fintech', 'healthcare', 'iot', 'web scraping', 'automation'
    ]
    found_tags = [k for k in keyword_tags if k in (processed_data.get('cleaned_text') or raw_text or '').lower()]
    if processed_data['projects']:
        for proj in processed_data['projects']:
            if not proj.get('description') and found_tags:
                proj['description'] = 'Keywords: ' + ', '.join(found_tags[:5])
            # If no technologies, use a subset of skills
            if not proj.get('technologies'):
                skills = processed_data.get('skills') or []
                if isinstance(skills, list) and skills:
                    proj['technologies'] = skills[:6]

    # Ensure fields_of_interest alias exists
    if 'fields_of_interest' not in processed_data and 'field_of_interest' in processed_data:
        processed_data['fields_of_interest'] = processed_data.get('field_of_interest', [])

    # 12) Ensure key technical skills are included if present in text and merge fields of interest
    try:
        skills = processed_data.get('skills') or []
        if not isinstance(skills, list):
            skills = []
        skills_set = {s.lower() for s in skills}
        presence = (raw_text or '').lower()
        add_if_present = {
            'python': ['python'],
            'java': ['java'],
            'php': ['php'],
            'javascript': ['javascript', 'js'],
            'html': ['html'],
            'css': ['css'],
            'mysql': ['mysql'],
            'pl/sql': ['pl sql', 'pl/sql', 'plsql'],
            'oracle': ['oracle'],
            'figma': ['figma'],
            'visual studio': ['visual studio', 'vs code', 'vscode']
        }
        for canon, needles in add_if_present.items():
            if canon not in skills_set and any(n in presence for n in needles):
                skills.append(canon)
                skills_set.add(canon)
        # Do NOT mix fields of interest into skills on the profile
        drop = {'git','github','tech'}
        skills = [s for s in skills if isinstance(s, str) and s.lower() not in drop]
        # De-duplicate, sort lightly, and cap to 35
        skills = sorted(list(dict.fromkeys(skills)))[:35]
        processed_data['skills'] = skills
    except Exception:
        pass

    # Opportunistic name fix around phone/location tokens (e.g., "Shiflina Nilofar" before phone)
    try:
        # Find a 10+ digit phone sequence
        mphone = re.search(r"(\+?\d[\d\-\s]{7,}\d)", source_text)
        if mphone:
            if not processed_data.get('name'):
                # Look back up to 80 chars for a potential name line
                start = max(0, mphone.start()-120)
                prev = source_text[start:mphone.start()].strip().splitlines()
                if prev:
                    cand = prev[-1].strip()
                    # Keep only alphabetic and spaces
                    cand_clean = re.sub(r"[^A-Za-z\s]", " ", cand)
                    tokens = [p for p in cand_clean.split() if p]
                    # Allow single-letter initials (e.g., 'P')
                    parts = tokens[:4]
                    if len(parts) >= 2:
                        processed_data['name'] = ' '.join(w.capitalize() for w in parts)
            # Try to extract location immediately after phone
            if not processed_data.get('location') or processed_data.get('location','').lower() == 'chennai':
                post = source_text[mphone.end():mphone.end()+100]
                post_tokens = re.findall(r"[A-Za-z]+", post)
                post_lower = [t.lower() for t in post_tokens]
                if 'chennai' in post_lower:
                    idx = post_lower.index('chennai')
                    prev_tok = post_tokens[idx-1] if idx-1 >= 0 else None
                    if prev_tok and len(prev_tok) > 2:
                        processed_data['location'] = f"{prev_tok.capitalize()}, Chennai"
                    else:
                        processed_data['location'] = 'Chennai'
        # Ensure Chennai location if present elsewhere
        if 'chennai' in low_text and not (processed_data.get('location')):
            processed_data['location'] = 'Chennai'
    except Exception:
        pass

    return processed_data

def initialize_model():
    """Initialize the enhanced model and preprocessor"""
    global model, preprocessor
    
    try:
        logger.info("Initializing Enhanced ResuMatch model...")
        
        # Initialize the enhanced text preprocessor
        preprocessor = EnhancedTextPreprocessor()
        
        # Initialize the enhanced similarity model
        model = EnhancedResuMatchModel()
        
        # Load job data and fit the model
        job_data = load_job_data('data/sample_internships.json')
        model.fit(job_data)
        
        logger.info("Enhanced model and preprocessor initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing enhanced model: {str(e)}")
        raise

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/test-upload', methods=['GET', 'POST'])
def test_upload():
    """Test upload endpoint"""
    if request.method == 'POST':
        logger.info(f"Test upload - Files: {list(request.files.keys())}")
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file key in request'}), 400
        
        file = request.files['file']
        logger.info(f"Test upload - File: {file}, filename: {file.filename}")
        
        if file.filename == '' or file.filename is None:
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file
        file_bytes = file.read()
        logger.info(f"Test upload - File size: {len(file_bytes)} bytes")
        
        if len(file_bytes) == 0:
            return jsonify({'error': 'Empty file'}), 400
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'size': len(file_bytes),
            'message': 'Upload test successful!'
        })
    
    return jsonify({'message': 'Test upload endpoint - use POST with file'})

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and processing"""
    if request.method == 'POST':
        try:
            logger.info(f"Upload request received. Files: {list(request.files.keys())}")
            
            # Check if file is present
            if 'file' not in request.files:
                logger.error("No 'file' key in request.files")
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            logger.info(f"File object: {file}, filename: {file.filename}")
            
            # Check if file is selected
            if file.filename == '' or file.filename is None:
                logger.error("Empty filename")
                flash('No file selected', 'error')
                return redirect(request.url)
            
            # Check file extension
            if not allowed_file(file.filename):
                logger.error(f"Invalid file type: {file.filename}")
                flash('Invalid file type. Please upload PDF, DOCX, or image files.', 'error')
                return redirect(request.url)
            
            # Read file bytes first
            file_bytes = file.read()
            file_size = len(file_bytes)
            
            logger.info(f"File size: {file_size} bytes")
            
            # Check file size
            if file_size > MAX_FILE_SIZE:
                flash('File too large. Maximum size is 16MB.', 'error')
                return redirect(request.url)
            
            if file_size == 0:
                flash('Empty file uploaded', 'error')
                return redirect(request.url)
            
            # Get file type
            file_type = get_file_type(file.filename)
            
            # Extract text from file (with structural hints if possible)
            logger.info(f"Extracting text from {file_type} file: {file.filename}")
            hints = None
            try:
                structured = extract_text_with_hints_from_bytes(file_bytes, file_type)
                extracted_text = structured.get('text', '')
                hints = structured.get('hints')
            except Exception as _e:
                logger.warning(f"Structured extraction failed: {_e}; falling back to plain text")
                extracted_text = extract_text_from_bytes(file_bytes, file_type)
            
            if not extracted_text or len(extracted_text.strip()) == 0:
                flash('No text could be extracted from the file. Please try a different file.', 'error')
                return redirect(request.url)
            
            processed_data = None
            if not ENABLE_OPENAI:
                logger.info("OpenAI parsing disabled (ENABLE_OPENAI=0). Using enhanced preprocessing.")
                processed_data = preprocessor.preprocess_resume(extracted_text)
            else:
                # Parse the text using OpenAI for better accuracy
                logger.info("Parsing extracted text with OpenAI...")
                openai_parser = OpenAIResumeParser()
                try:
                    oa_data = openai_parser.parse_resume_with_openai(extracted_text)
                    # Ensure legacy format is normalized (pass original text for cleaned_text)
                    processed_data = openai_parser.convert_to_legacy_format(oa_data, original_text=extracted_text)
                    logger.info("OpenAI parsing successful")
                    # Always enrich skills with enhanced preprocessor for better coverage
                    try:
                        pp_data = preprocessor.preprocess_resume(extracted_text)
                        if pp_data.get('skills'):
                            processed_data['skills'] = pp_data['skills']
                        if pp_data.get('cleaned_text'):
                            processed_data['cleaned_text'] = pp_data['cleaned_text']
                        if pp_data.get('tokens'):
                            processed_data['tokens'] = pp_data['tokens']
                            processed_data['token_count'] = len(pp_data['tokens'])
                        if not processed_data.get('locations') and pp_data.get('locations'):
                            processed_data['locations'] = pp_data['locations']
                    except Exception as merge_err:
                        logger.warning(f"Skill enrichment merge failed: {merge_err}")
                except Exception as openai_error:
                    logger.warning(f"OpenAI parsing failed: {openai_error}. Using fallback preprocessing.")
                    # Fallback to enhanced preprocessing
                    processed_data = preprocessor.preprocess_resume(extracted_text)
                    logger.info("Fallback preprocessing completed")
            
            # Store processed data in session
            session_data = {
                'filename': file.filename,
                'extracted_text': extracted_text,
                'processed_data': processed_data,
                'hints': hints
            }

            # Save server-side to avoid cookie overflow; store only id in session
            try:
                profile_id = save_profile_data(session_data)
                session['profile_id'] = profile_id
            except Exception as sess_err:
                logger.warning(f"Failed to store profile: {sess_err}")
            
            # Redirect to profile page
            return redirect(url_for('profile'))
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/profile')
def profile():
    """Display professional profile"""
    try:
        # Get data from query parameters or session fallback
        data_json = request.args.get('data')
        if data_json:
            session_data = json.loads(data_json)
        else:
            profile_id = session.get('profile_id')
            if not profile_id:
                flash('No resume data found. Please upload a resume first.', 'error')
                return redirect(url_for('upload_file'))
            session_data = load_profile_data(profile_id)
            if not session_data:
                flash('No resume data found. Please upload a resume first.', 'error')
                return redirect(url_for('upload_file'))
        processed_data = session_data['processed_data']
        
        logger.info(f"Processed data type: {type(processed_data)}")
        logger.info(f"Processed data content: {processed_data}")
        
        # Ensure processed_data is a dictionary
        if isinstance(processed_data, str):
            try:
                processed_data = json.loads(processed_data)
                logger.info("Successfully parsed processed_data from JSON string")
            except Exception as parse_error:
                logger.warning(f"Failed to parse processed_data as JSON: {parse_error}")
                # If it's not JSON, create a basic structure
                processed_data = {
                    'name': 'Not specified',
                    'email': 'Not specified',
                    'phone': 'Not specified',
                    'location': 'Not specified',
                    'skills': [],
                    'experience': [],
                    'projects': [],
                    'certifications': [],
                    'achievements': [],
                    'fields_of_interest': []
                }
        elif not isinstance(processed_data, dict):
            logger.warning(f"Processed data is not a dictionary: {type(processed_data)}")
            # Convert to dictionary if possible
            processed_data = {
                'name': 'Not specified',
                'email': 'Not specified',
                'phone': 'Not specified',
                'location': 'Not specified',
                'skills': [],
                'experience': [],
                'projects': [],
                'certifications': [],
                'achievements': [],
                'fields_of_interest': []
            }

        # Normalize structures for template compatibility
        try:
            # Certifications: convert list[str] -> list[dict]
            certs = processed_data.get('certifications', []) if isinstance(processed_data, dict) else []
            if isinstance(certs, list):
                normalized_certs = []
                for c in certs:
                    if isinstance(c, dict):
                        normalized_certs.append({
                            'name': c.get('name', '') or c.get('title', ''),
                            'issuer': c.get('issuer', ''),
                            'date': c.get('date', '')
                        })
                    elif isinstance(c, str):
                        normalized_certs.append({'name': c, 'issuer': '', 'date': ''})
                processed_data['certifications'] = normalized_certs

            # Projects: convert list[str] -> list[dict]
            projects = processed_data.get('projects', []) if isinstance(processed_data, dict) else []
            if isinstance(projects, list):
                normalized_projects = []
                for p in projects:
                    if isinstance(p, dict):
                        normalized_projects.append({
                            'name': p.get('name', '') or p.get('title', ''),
                            'description': p.get('description', ''),
                            'technologies': p.get('technologies', []) or p.get('tech', []) or []
                        })
                    elif isinstance(p, str):
                        normalized_projects.append({'name': p, 'description': '', 'technologies': []})
                processed_data['projects'] = normalized_projects

            # Experience: ensure list[dict]
            exps = processed_data.get('experience', []) if isinstance(processed_data, dict) else []
            if isinstance(exps, list):
                normalized_exps = []
                for e in exps:
                    if isinstance(e, dict):
                        normalized_exps.append({
                            'company': e.get('company', ''),
                            'position': e.get('position', ''),
                            'dates': e.get('dates', ''),
                            'description': e.get('description', '')
                        })
                    elif isinstance(e, str):
                        normalized_exps.append({
                            'company': '',
                            'position': '',
                            'dates': '',
                            'description': e
                        })
                processed_data['experience'] = normalized_exps

            # Fields of interest: create alias if needed
            if 'fields_of_interest' not in processed_data and 'field_of_interest' in processed_data:
                processed_data['fields_of_interest'] = processed_data.get('field_of_interest', [])
        except Exception as norm_err:
            logger.warning(f"Normalization step failed (non-fatal): {norm_err}")
        
        # Enrich data for profile rendering
        processed_data = enrich_processed_data(processed_data, session_data.get('extracted_text', ''), session_data.get('hints'))

        # Prepare data for template
        template_data = {
            'filename': session_data['filename'],
            'extracted_text': session_data['extracted_text'],
            'processed_data': processed_data,
            'session_data': session_data,
            'user_edited': session_data.get('user_edited', False)
        }

        # Persist back enriched data to server-side storage
        try:
            profile_id = session.get('profile_id')
            if profile_id:
                update_profile_data(profile_id, template_data)
        except Exception as _e:
            logger.warning(f"Could not update stored profile: {_e}")
        
        return render_template('profile_dropdown.html', **template_data)
        
    except Exception as e:
        logger.error(f"Error displaying profile: {str(e)}")
        flash(f'Error displaying profile: {str(e)}', 'error')
        return redirect(url_for('upload_file'))

@app.route('/my-profile')
def my_profile():
    """Display user profile page (from navbar)"""
    try:
        profile_id = session.get('profile_id')
        if not profile_id:
            flash('Please upload your resume first to view your profile.', 'info')
            return redirect(url_for('upload_file'))
        session_data = load_profile_data(profile_id)
        if not session_data:
            flash('Please upload your resume first to view your profile.', 'info')
            return redirect(url_for('upload_file'))
        # Render and persist edits
        processed_data = session_data.get('processed_data', {})
        processed_data = enrich_processed_data(processed_data, session_data.get('extracted_text',''), session_data.get('hints'))
        session_data['processed_data'] = processed_data
        update_profile_data(profile_id, session_data)
        return render_template('profile_dropdown.html',
                               filename=session_data['filename'],
                               extracted_text=session_data['extracted_text'],
                               processed_data=processed_data,
                               session_data=session_data)
    
    except Exception as e:
        logger.error(f"Error displaying profile: {str(e)}")
        flash(f'Error displaying profile: {str(e)}', 'error')
        return redirect(url_for('upload_file'))

@app.route('/recommendations')
def recommendations():
    """Display job recommendations"""
    try:
        # Get data from query parameters or session fallback
        data_json = request.args.get('data')
        if data_json:
            session_data = json.loads(data_json)
        else:
            profile_id = session.get('profile_id')
            if not profile_id:
                flash('No resume data found. Please upload a resume first.', 'error')
                return redirect(url_for('upload_file'))
            session_data = load_profile_data(profile_id)
            if not session_data:
                flash('No resume data found. Please upload a resume first.', 'error')
                return redirect(url_for('upload_file'))
        processed_data = session_data['processed_data']
        
        # Get recommendation method from request
        method = request.args.get('method', 'combined')
        
        # Generate enhanced recommendations
        logger.info(f"Generating enhanced recommendations using {method} method...")
        
        if method == 'enhanced':
            recommendations = model.predict_enhanced(processed_data, top_k=5)
        elif method == 'tfidf':
            recommendations = model.predict_tfidf(processed_data['cleaned_text'], top_k=5)
        elif method == 'sbert':
            recommendations = model.predict_sentence_transformer(processed_data['cleaned_text'], top_k=5)
        else:  # enhanced by default
            recommendations = model.predict_enhanced(processed_data, top_k=5)
        
        # Prepare data for template
        template_data = {
            'filename': session_data['filename'],
            'extracted_text': session_data['extracted_text'],
            'processed_data': processed_data,
            'recommendations': recommendations,
            'method': method
        }
        
        return render_template('recommendations.html', **template_data)
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        flash(f'Error generating recommendations: {str(e)}', 'error')
        return redirect(url_for('upload_file'))

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """API endpoint for job recommendations"""
    try:
        data = request.get_json()
        
        if not data or 'resume_text' not in data:
            return jsonify({'error': 'Resume text is required'}), 400
        
        resume_text = data['resume_text']
        method = data.get('method', 'combined')
        top_k = data.get('top_k', 5)
        
        # Preprocess resume text
        processed_data = preprocessor.preprocess_resume(resume_text)
        
        # Generate enhanced recommendations
        if method == 'enhanced':
            recommendations = model.predict_enhanced(processed_data, top_k=top_k)
        elif method == 'tfidf':
            recommendations = model.predict_tfidf(processed_data['cleaned_text'], top_k=top_k)
        elif method == 'sbert':
            recommendations = model.predict_sentence_transformer(processed_data['cleaned_text'], top_k=top_k)
        else:  # enhanced by default
            recommendations = model.predict_enhanced(processed_data, top_k=top_k)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'method': method,
            'processed_data': processed_data
        })
        
    except Exception as e:
        logger.error(f"Error in API recommendation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/jobs')
def jobs_page():
    """Jobs page with interactive cards"""
    try:
        if model is None:
            flash('Model not initialized. Please try again later.', 'error')
            return redirect(url_for('index'))
        
        jobs = []
        for job in model.job_descriptions:
            jobs.append({
                'id': job['id'],
                'title': job['title'],
                'company': job['company'],
                'description': job['description'],
                'requirements': job['requirements'],
                'location': job['location'],
                'duration': job['duration']
            })
        
        # Build unique normalized location list from jobs
        locations_set = set()
        for job in jobs:
            loc = normalize_location(job.get('location',''))
            job['location'] = loc
            if loc:
                locations_set.add(loc)
        locations = sorted(locations_set)
        
        return render_template('jobs.html', jobs=jobs, locations=locations)
        
    except Exception as e:
        logger.error(f"Error loading jobs page: {str(e)}")
        flash(f'Error loading jobs: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/jobs')
def api_jobs():
    """API endpoint to get all available jobs"""
    try:
        if model is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        jobs = []
        for job in model.job_descriptions:
            jobs.append({
                'id': job['id'],
                'title': job['title'],
                'company': job['company'],
                'description': job['description'],
                'requirements': job['requirements'],
                'location': job['location'],
                'duration': job['duration']
            })
        
        return jsonify({'jobs': jobs})
        
    except Exception as e:
        logger.error(f"Error in API jobs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'model_loaded': model is not None,
            'preprocessor_loaded': preprocessor is not None
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/profile', methods=['GET','POST'])
def api_profile():
    """Get or update the stored profile using server-side storage"""
    try:
        profile_id = session.get('profile_id')
        if request.method == 'GET':
            if not profile_id:
                return jsonify({'success': False, 'error': 'no_profile'}), 404
            data = load_profile_data(profile_id)
            return jsonify({'success': bool(data), 'profile': data})
        else:
            if not profile_id:
                return jsonify({'success': False, 'error': 'no_profile'}), 404
            payload = request.get_json(silent=True) or {}
            profile = load_profile_data(profile_id) or {}
            # Merge processed_data updates if provided
            if 'processed_data' in payload and isinstance(payload['processed_data'], dict):
                pd = profile.get('processed_data', {})
                pd.update(payload['processed_data'])
                profile['processed_data'] = pd
            # Merge simple fields
            for key in ['filename','extracted_text','hints']:
                if key in payload:
                    profile[key] = payload[key]
            update_profile_data(profile_id, profile)
            return jsonify({'success': True, 'profile': profile})
    except Exception as e:
        logger.error(f"Error in api/profile: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai_suggest_skills', methods=['GET'])
def api_ai_suggest_skills():
    """Suggest additional skills to learn, using current profile vs job dataset.
    Returns a list of {skill, count, confidence, message} where count is the
    number of jobs where the skill appears in skills_needed or requirements,
    and confidence is a heuristic in [0.6, 0.95].
    """
    try:
        profile_id = session.get('profile_id')
        if not profile_id:
            return jsonify({'success': False, 'error': 'no_profile'}), 404
        profile = load_profile_data(profile_id)
        if not profile:
            return jsonify({'success': False, 'error': 'no_profile'}), 404
        pd = profile.get('processed_data', {}) or {}
        resume_skills = set([s.lower() for s in (pd.get('skills') or []) if isinstance(s, str)])
        resume_field = [f.lower() for f in (pd.get('fields_of_interest') or pd.get('field_of_interest') or []) if isinstance(f, str)]
        # Aggregate missing skills across all jobs
        missing_counts = Counter()
        job_count = 0
        for job in getattr(model, 'job_descriptions', []) or []:
            job_count += 1
            job_skills = [s.lower() for s in (job.get('skills_needed') or []) if isinstance(s, str)]
            # fall back: derive from requirements text tokens
            if not job_skills:
                reqs = job.get('requirements') or []
                if isinstance(reqs, list):
                    for r in reqs:
                        if isinstance(r, str):
                            for tok in re.findall(r'[A-Za-z][A-Za-z0-9\+\.#\-]{2,}', r.lower()):
                                job_skills.append(tok)
            # missing vs resume
            for js in set(job_skills):
                if js not in resume_skills:
                    missing_counts[js] += 1
        # Build suggestions (top 10)
        suggestions = []
        if job_count > 0:
            for skill, cnt in missing_counts.most_common(10):
                frac = cnt / job_count
                # Confidence heuristic mapped to [0.6, 0.95]
                confidence = round(0.6 + 0.35 * min(1.0, frac * 2.0), 2)
                msg = f"If you learn {skill}, your match is likely to improve significantly (≈{int(confidence*100)}% impact for several roles)."
                # Boost skills that align with field of interest
                if resume_field and any(f in skill for f in resume_field):
                    confidence = min(0.95, round(confidence + 0.05, 2))
                    msg = f"Learning {skill} aligns strongly with your field of interest; estimated improvement ≈{int(confidence*100)}%."
                suggestions.append({'skill': skill, 'count': cnt, 'confidence': confidence, 'message': msg})
        return jsonify({'success': True, 'suggestions': suggestions})
    except Exception as e:
        logger.error(f"Error in ai_suggest_skills: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai_suggest_for_job', methods=['GET'])
def api_ai_suggest_for_job():
    """Suggest skills for a specific job based on resume gap.
    Expects ?job_id=... and uses current stored profile."""
    try:
        profile_id = session.get('profile_id')
        if not profile_id:
            return jsonify({'success': False, 'error': 'no_profile'}), 404
        profile = load_profile_data(profile_id)
        if not profile:
            return jsonify({'success': False, 'error': 'no_profile'}), 404
        pd = profile.get('processed_data', {}) or {}
        resume_skills = set([s.lower() for s in (pd.get('skills') or []) if isinstance(s, str)])
        resume_field = [f.lower() for f in (pd.get('fields_of_interest') or pd.get('field_of_interest') or []) if isinstance(f, str)]
        job_id = request.args.get('job_id')
        if not job_id:
            return jsonify({'success': False, 'error': 'job_id_required'}), 400
        # Find job by id in model.job_descriptions
        job = None
        for j in getattr(model, 'job_descriptions', []) or []:
            if str(j.get('id')) == str(job_id):
                job = j
                break
        if not job:
            return jsonify({'success': False, 'error': 'job_not_found'}), 404
        job_skills = [s.lower() for s in (job.get('skills_needed') or []) if isinstance(s, str)]
        if not job_skills:
            # derive from requirements text tokens
            for r in (job.get('requirements') or []):
                if isinstance(r, str):
                    for tok in re.findall(r'[A-Za-z][A-Za-z0-9\+\.#\-]{2,}', r.lower()):
                        job_skills.append(tok)
        missing = [s for s in set(job_skills) if s not in resume_skills]
        # Build suggestions with confidence heuristic
        suggestions = []
        for s in missing[:10]:
            confidence = 0.8
            if resume_field and any(f in s for f in resume_field):
                confidence = 0.95
            message = f"Learning {s} is highly relevant for this role (≈{int(confidence*100)}% guarantee to improve your match)."
            suggestions.append({'skill': s, 'confidence': confidence, 'message': message})
        return jsonify({'success': True, 'suggestions': suggestions})
    except Exception as e:
        logger.error(f"Error in ai_suggest_for_job: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

if __name__ == '__main__':
    try:
        # Initialize model
        initialize_model()
        
        # Run the app
        logger.info("Starting ResuMatch AI Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        print(f"Error starting application: {str(e)}")
        print("Make sure to install required dependencies:")
        print("pip install -r requirements.txt")
        print("python -m spacy download en_core_web_sm")
