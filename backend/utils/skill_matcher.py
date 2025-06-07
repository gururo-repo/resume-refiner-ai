import re
from typing import List, Dict, Set
from difflib import SequenceMatcher
import json
import os
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from .model_loader import get_models, get_match_model, get_feature_scaler, get_model_features, get_sentence_transformer
from .genai_suggester import ResumeImprover

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

class SkillMatcher:
    def __init__(self):
        try:
            self.models = get_models()
            self.match_model = get_match_model()
            self.feature_scaler = get_feature_scaler()
            self.model_features = get_model_features()
            self.sentence_transformer = get_sentence_transformer()
            logger.info("Successfully initialized SkillMatcher")
        except Exception as e:
            logger.error(f"Failed to initialize SkillMatcher: {str(e)}")
            raise

    def extract_and_match_skills(self, resume_text, job_description):
        """
        Extract and match skills between resume and job description.
        First attempts LLM-based analysis, falls back to model-based analysis if LLM fails.
        """
        try:
            # First attempt: Use LLM-based analysis
            suggester = ResumeImprover()
            analysis = suggester.analyze_skills(resume_text, job_description)
            
            # Extract skills from LLM analysis
            matched_skills = analysis.get('matched_skills', [])
            missing_skills = analysis.get('missing_skills', [])
            additional_skills = analysis.get('additional_skills', [])
            
            logger.info("Successfully completed LLM-based skill analysis")
            return {
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
                'additional_skills': additional_skills,
                'analysis_type': 'llm'
            }
            
        except Exception as e:
            logger.warning(f"LLM analysis failed, falling back to model-based analysis: {str(e)}")
            return self._model_based_skill_analysis(resume_text, job_description)

    def _model_based_skill_analysis(self, resume_text, job_description):
        """
        Fallback method using trained model for skill analysis.
        Only used when LLM analysis fails.
        """
        try:
            # Extract skills from both texts
            resume_skills = self._extract_skills_from_text(resume_text)
            job_skills = self._extract_skills_from_text(job_description)
            
            # Calculate matched, missing, and additional skills
            matched_skills = list(resume_skills.intersection(job_skills))
            missing_skills = list(job_skills - resume_skills)
            additional_skills = list(resume_skills - job_skills)
            
            logger.info("Successfully completed model-based skill analysis")
            return {
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
                'additional_skills': additional_skills,
                'analysis_type': 'model'
            }
            
        except Exception as e:
            logger.error(f"Model-based skill analysis failed: {str(e)}")
            return {
                'matched_skills': [],
                'missing_skills': [],
                'additional_skills': [],
                'analysis_type': 'fallback'
            }

    def _extract_skills_from_text(self, text):
        """
        Extract skills from text using semantic similarity.
        """
        try:
            # Get text embedding
            text_embedding = self.sentence_transformer.encode(text)
            
            # Compare with known skill embeddings
            skills = set()
            for skill, embedding in self.model_features.items():
                similarity = cosine_similarity([text_embedding], [embedding])[0][0]
                if similarity > 0.5:  # Threshold for skill extraction
                    skills.add(skill)
            
            return skills
            
        except Exception as e:
            logger.error(f"Failed to extract skills from text: {str(e)}")
            return set()

# Create a singleton instance
skill_matcher = SkillMatcher()

def extract_and_match_skills(resume_text, job_description):
    """
    Extract and match skills between resume and job description.
    This is the main entry point for skill matching.
    """
    return skill_matcher.extract_and_match_skills(resume_text, job_description) 