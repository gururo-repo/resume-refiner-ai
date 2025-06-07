import re
from typing import List, Dict, Set
from difflib import SequenceMatcher
import json
import os

# Load comprehensive skill database
SKILLS_DB = {
    'programming': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
        'go', 'rust', 'scala', 'perl', 'r', 'matlab', 'sql', 'nosql', 'mongodb', 'postgresql',
        'mysql', 'oracle', 'sqlite'
    ],
    'frameworks': [
        'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'node.js', 'laravel',
        'ruby on rails', 'asp.net', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas',
        'numpy', 'bootstrap', 'tailwind', 'material-ui'
    ],
    'cloud': [
        'aws', 'azure', 'gcp', 'cloud', 's3', 'ec2', 'lambda', 'dynamodb', 'rds', 'cloudfront',
        'route53', 'vpc', 'iam', 'kubernetes', 'docker', 'terraform', 'ansible', 'jenkins',
        'ci/cd', 'devops'
    ],
    'data_science': [
        'machine learning', 'deep learning', 'ai', 'artificial intelligence', 'nlp', 'computer vision',
        'data analysis', 'data visualization', 'statistics', 'big data', 'hadoop', 'spark',
        'tableau', 'power bi', 'excel', 'r', 'python'
    ],
    'soft_skills': [
        'leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking',
        'project management', 'agile', 'scrum', 'time management', 'adaptability',
        'creativity', 'collaboration', 'presentation', 'negotiation'
    ]
}

def normalize_text(text: str) -> str:
    """Normalize text for better matching."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_skill_similarity(skill1: str, skill2: str) -> float:
    """Calculate similarity between two skills using SequenceMatcher."""
    return SequenceMatcher(None, skill1.lower(), skill2.lower()).ratio()

def extract_skills(text: str, threshold: float = 0.8) -> Dict[str, List[str]]:
    """Extract skills from text using fuzzy matching."""
    normalized_text = normalize_text(text)
    found_skills = {category: [] for category in SKILLS_DB.keys()}
    
    for category, skills in SKILLS_DB.items():
        for skill in skills:
            # Check for exact match
            if skill.lower() in normalized_text:
                found_skills[category].append(skill)
                continue
            
            # Check for fuzzy match
            words = normalized_text.split()
            for word in words:
                if len(word) > 3 and get_skill_similarity(skill, word) > threshold:
                    found_skills[category].append(skill)
                    break
    
    return found_skills

def get_missing_skills(resume_skills: Dict[str, List[str]], 
                      jd_skills: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Find missing skills by comparing resume and JD skills."""
    missing_skills = {category: [] for category in SKILLS_DB.keys()}
    
    for category in SKILLS_DB.keys():
        resume_category_skills = set(resume_skills.get(category, []))
        jd_category_skills = set(jd_skills.get(category, []))
        missing_skills[category] = list(jd_category_skills - resume_category_skills)
    
    return missing_skills

def get_skill_gaps(resume_text: str, jd_text: str) -> Dict:
    """Analyze skill gaps between resume and job description."""
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    missing_skills = get_missing_skills(resume_skills, jd_skills)
    
    # Calculate skill match percentage
    total_jd_skills = sum(len(skills) for skills in jd_skills.values())
    total_missing = sum(len(skills) for skills in missing_skills.values())
    match_percentage = ((total_jd_skills - total_missing) / total_jd_skills * 100) if total_jd_skills > 0 else 0
    
    return {
        'resume_skills': resume_skills,
        'jd_skills': jd_skills,
        'missing_skills': missing_skills,
        'match_percentage': round(match_percentage, 2),
        'skill_gaps': {
            category: {
                'missing': skills,
                'priority': 'high' if category in ['programming', 'frameworks'] else 'medium'
            }
            for category, skills in missing_skills.items()
            if skills
        }
    }

def extract_and_match_skills(resume_text: str, jd_text: str) -> Dict:
    """Main function to extract and match skills between resume and JD."""
    return get_skill_gaps(resume_text, jd_text) 