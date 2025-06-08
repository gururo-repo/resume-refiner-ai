import re
from typing import List, Dict, Set
from difflib import SequenceMatcher
import json
import os
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from .model_loader import get_models, get_match_model, get_feature_scaler, get_model_features, get_sentence_transformer, get_model_loader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load comprehensive skill database
SKILLS_DB = {
    'programming': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
        'go', 'rust', 'scala', 'perl', 'r', 'matlab', 'sql', 'nosql', 'mongodb', 'postgresql',
        'mysql', 'oracle', 'sqlite', 'graphql', 'rest api', 'soap', 'xml', 'json', 'yaml',
        'bash', 'powershell', 'shell scripting', 'assembly', 'fortran', 'cobol'
    ],
    'frameworks': [
        'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'node.js', 'laravel',
        'ruby on rails', 'asp.net', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas',
        'numpy', 'bootstrap', 'tailwind', 'material-ui', 'next.js', 'nuxt.js', 'gatsby',
        'fastapi', 'fastify', 'nest.js', 'jquery', 'redux', 'mobx', 'vuex', 'graphql',
        'apollo', 'jest', 'mocha', 'chai', 'cypress', 'selenium', 'pytest', 'junit'
    ],
    'cloud': [
        'aws', 'azure', 'gcp', 'cloud', 's3', 'ec2', 'lambda', 'dynamodb', 'rds', 'cloudfront',
        'route53', 'vpc', 'iam', 'kubernetes', 'docker', 'terraform', 'ansible', 'jenkins',
        'ci/cd', 'devops', 'elastic beanstalk', 'cloudformation', 'cloudwatch', 'sns', 'sqs',
        'api gateway', 'cognito', 'cloudtrail', 'waf', 'shield', 'cloudfront', 'elasticache',
        'redshift', 'aurora', 'neptune', 'documentdb', 'opensearch', 'elasticsearch',
        'eks', 'ecs', 'fargate', 'app runner', 'lambda', 'step functions', 'eventbridge'
    ],
    'data_science': [
        'machine learning', 'deep learning', 'ai', 'artificial intelligence', 'nlp', 'computer vision',
        'data analysis', 'data visualization', 'statistics', 'big data', 'hadoop', 'spark',
        'tableau', 'power bi', 'excel', 'r', 'python', 'pandas', 'numpy', 'scipy',
        'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'opencv', 'nltk', 'spacy',
        'bert', 'gpt', 'transformers', 'xgboost', 'lightgbm', 'catboost', 'h2o',
        'apache spark', 'apache kafka', 'apache flink', 'apache beam', 'apache airflow',
        'dbt', 'snowflake', 'databricks', 'looker', 'qlik', 'metabase', 'superset'
    ],
    'devops': [
        'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions', 'circleci',
        'travis ci', 'terraform', 'ansible', 'puppet', 'chef', 'prometheus', 'grafana',
        'elk stack', 'splunk', 'datadog', 'new relic', 'nagios', 'zabbix', 'consul',
        'vault', 'istio', 'linkerd', 'helm', 'argo', 'flux', 'spinnaker', 'rancher',
        'aws codepipeline', 'azure devops', 'gcp cloud build'
    ],
    'security': [
        'security', 'cybersecurity', 'penetration testing', 'vulnerability assessment',
        'security compliance', 'gdpr', 'hipaa', 'pci dss', 'iso 27001', 'nist',
        'owasp', 'security architecture', 'network security', 'application security',
        'cloud security', 'devsecops', 'siem', 'soar', 'waf', 'ids', 'ips',
        'firewall', 'vpn', 'encryption', 'authentication', 'authorization', 'iam',
        'zero trust', 'threat modeling', 'risk assessment'
    ],
    'soft_skills': [
        'leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking',
        'project management', 'agile', 'scrum', 'time management', 'adaptability',
        'creativity', 'collaboration', 'presentation', 'negotiation', 'mentoring',
        'coaching', 'conflict resolution', 'emotional intelligence', 'decision making',
        'strategic thinking', 'innovation', 'customer focus', 'business acumen',
        'stakeholder management', 'change management', 'risk management'
    ],
    'tools': [
        'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'trello',
        'asana', 'slack', 'microsoft teams', 'zoom', 'postman', 'swagger',
        'vscode', 'intellij', 'eclipse', 'vim', 'emacs', 'sublime text',
        'atom', 'notepad++', 'figma', 'sketch', 'adobe xd', 'invision',
        'zeplin', 'maven', 'gradle', 'npm', 'yarn', 'pip', 'conda',
        'docker compose', 'kubernetes dashboard', 'kubectl', 'helm'
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

def extract_skills(text):
    """Extract skills from text using the skills database."""
    try:
        if not text:
            return {}
            
        text = text.lower()
        skills_by_category = {}
        
        # Initialize categories
        for category in SKILLS_DB.keys():
            skills_by_category[category] = set()
        
        # Extract skills from each category
        for category, skills in SKILLS_DB.items():
            for skill in skills:
                if skill.lower() in text:
                    skills_by_category[category].add(skill)
        
        # Convert sets to lists for JSON serialization
        return {category: list(skills) for category, skills in skills_by_category.items()}
        
    except Exception as e:
        logger.error(f"Failed to extract skills from text: {str(e)}")
        return {}

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

class SkillMatcher:
    def __init__(self):
        """Initialize the SkillMatcher with common technical skills."""
        self.model_loader = get_model_loader()
        self.common_skills = {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php',
            'swift', 'kotlin', 'go', 'rust', 'scala', 'perl', 'r', 'matlab',
            
            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django',
            'flask', 'spring', 'asp.net', 'laravel', 'symfony', 'jquery', 'bootstrap',
            'tailwind', 'sass', 'less', 'webpack', 'babel', 'npm', 'yarn',
            
            # Databases
            'sql', 'mysql', 'postgresql', 'oracle', 'sql server', 'mongodb', 'redis',
            'cassandra', 'elasticsearch', 'dynamodb', 'firebase', 'neo4j',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab',
            'github actions', 'terraform', 'ansible', 'puppet', 'chef', 'prometheus',
            'grafana', 'elk stack', 'splunk',
            
            # AI & ML
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
            'scikit-learn', 'numpy', 'pandas', 'opencv', 'nltk', 'spacy', 'bert',
            'gpt', 'computer vision', 'nlp', 'reinforcement learning',
            
            # Mobile Development
            'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic',
            
            # Testing
            'junit', 'pytest', 'selenium', 'cypress', 'jest', 'mocha', 'chai',
            'cucumber', 'jasmine', 'testng',
            
            # Other
            'git', 'agile', 'scrum', 'kanban', 'jira', 'confluence', 'rest',
            'graphql', 'microservices', 'serverless', 'ci/cd', 'security',
            'networking', 'linux', 'windows', 'macos'
        }
        logger.info("Successfully initialized SkillMatcher")
    
    def extract_required_skills(self, text: str) -> List[str]:
        """Extract required skills from text."""
        try:
            # Convert text to lowercase for case-insensitive matching
            text_lower = text.lower()
            
            # Find required skills using common patterns
            required_patterns = [
                r'required skills?:?\s*([^.]*)',
                r'must have:?\s*([^.]*)',
                r'requirements:?\s*([^.]*)',
                r'qualifications:?\s*([^.]*)'
            ]
            
            required_skills = set()
            for pattern in required_patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    section = match.group(1)
                    # Extract skills from the section
                    skills = self._extract_skills_from_section(section)
                    required_skills.update(skills)
            
            return list(required_skills)
            
        except Exception as e:
            logger.error(f"Error extracting required skills: {str(e)}")
            return []
    
    def extract_preferred_skills(self, text: str) -> List[str]:
        """Extract preferred skills from text."""
        try:
            # Convert text to lowercase for case-insensitive matching
            text_lower = text.lower()
            
            # Find preferred skills using common patterns
            preferred_patterns = [
                r'preferred skills?:?\s*([^.]*)',
                r'nice to have:?\s*([^.]*)',
                r'bonus:?\s*([^.]*)',
                r'additional skills?:?\s*([^.]*)'
            ]
            
            preferred_skills = set()
            for pattern in preferred_patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    section = match.group(1)
                    # Extract skills from the section
                    skills = self._extract_skills_from_section(section)
                    preferred_skills.update(skills)
            
            return list(preferred_skills)
            
        except Exception as e:
            logger.error(f"Error extracting preferred skills: {str(e)}")
            return []
    
    def _extract_skills_from_section(self, section: str) -> Set[str]:
        """Extract skills from a section of text."""
        try:
            # Split section into words and phrases
            words = section.split()
            phrases = []
            for i in range(len(words)):
                for j in range(i + 1, min(i + 4, len(words) + 1)):
                    phrases.append(' '.join(words[i:j]))
            
            # Find matching skills
            found_skills = set()
            for skill in self.common_skills:
                if skill in section.lower():
                    found_skills.add(skill)
            
            return found_skills
            
        except Exception as e:
            logger.error(f"Error extracting skills from section: {str(e)}")
            return set()

# Create a singleton instance
skill_matcher = SkillMatcher()

def extract_and_match_skills(resume_text, job_description):
    """
    Extract and match skills between resume and job description.
    This is the main entry point for skill matching.
    """
    try:
        # Extract skills from both texts
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)
        
        # Convert to sets for matching
        resume_skill_set = set()
        job_skill_set = set()
        
        for category, skills in resume_skills.items():
            resume_skill_set.update(skills)
            
        for category, skills in job_skills.items():
            job_skill_set.update(skills)
        
        # Find matching and missing skills
        matched_skills = list(resume_skill_set.intersection(job_skill_set))
        missing_skills = list(job_skill_set - resume_skill_set)
        
        return {
            'matched_skills': matched_skills,
            'missing_skills': missing_skills
        }
        
    except Exception as e:
        logger.error(f"Failed to match skills: {str(e)}")
        return {
            'matched_skills': [],
            'missing_skills': []
        } 