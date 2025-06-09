import os
import json
import numpy as np
import logging
from .genai_suggester import ResumeImprover
from .model_loader import get_model_loader
from .skill_matcher import extract_skills, SKILLS_DB
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, Tuple, List
import google.generativeai as genai
from dotenv import load_dotenv
from .groq_analyzer import get_groq_analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RolePredictor:
    """Predicts job roles based on resume and job description content."""
    
    def __init__(self):
        """Initialize the role predictor."""
        self.model_loader = get_model_loader()
        self.role_categories = {
            'software_engineering': [
                'software', 'developer', 'engineer', 'programming', 'coding',
                'frontend', 'backend', 'fullstack', 'web', 'mobile', 'application',
                'java', 'python', 'javascript', 'typescript', 'react', 'angular',
                'node.js', 'spring', 'django', 'flask', 'express'
            ],
            'data_science': [
                'data', 'analyst', 'scientist', 'machine learning', 'ai',
                'artificial intelligence', 'deep learning', 'statistics',
                'analytics', 'big data', 'data mining', 'predictive modeling',
                'python', 'r', 'sql', 'pandas', 'numpy', 'tensorflow', 'pytorch'
            ],
            'devops': [
                'devops', 'cloud', 'aws', 'azure', 'gcp', 'kubernetes',
                'docker', 'ci/cd', 'jenkins', 'gitlab', 'github actions',
                'terraform', 'ansible', 'puppet', 'chef', 'monitoring',
                'logging', 'infrastructure', 'automation'
            ],
            'product_management': [
                'product', 'manager', 'pm', 'agile', 'scrum', 'kanban',
                'jira', 'confluence', 'roadmap', 'backlog', 'sprint',
                'stakeholder', 'requirements', 'user stories', 'metrics',
                'analytics', 'strategy', 'vision'
            ],
            'design': [
                'design', 'ui', 'ux', 'graphic', 'creative', 'visual',
                'interaction', 'user experience', 'user interface', 'wireframe',
                'prototype', 'figma', 'sketch', 'adobe', 'photoshop',
                'illustrator', 'invision', 'research', 'usability'
            ]
        }

    def predict_role(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Predict the most likely role based on resume and job description.
        
        Args:
            resume_text: The resume text to analyze
            job_description: The job description text
            
        Returns:
            Dictionary containing role prediction and confidence
        """
        try:
            # First try Groq analysis
            groq_analysis = self.model_loader.try_groq_analysis(resume_text, job_description)
            if groq_analysis and 'role_analysis' in groq_analysis:
                logger.info("Using Groq role analysis")
                return groq_analysis['role_analysis']

            # If Groq fails, fall back to local models
            logger.info("Falling back to local model role prediction")
            
            # Get embeddings for both texts
            texts = [resume_text, job_description]
            embeddings = self.model_loader.get_embeddings(texts)
            
            if embeddings is not None:
                # Calculate similarity with each role category
                role_scores = {}
                for role, keywords in self.role_categories.items():
                    # Get embeddings for role keywords
                    keyword_embeddings = self.model_loader.get_embeddings(keywords)
                    if keyword_embeddings is not None:
                        # Calculate average similarity with role keywords
                        similarities = cosine_similarity(embeddings, keyword_embeddings)
                        role_scores[role] = float(np.mean(similarities))
                
                if role_scores:
                    # Get the role with highest score
                    primary_role = max(role_scores.items(), key=lambda x: x[1])
                    confidence = primary_role[1] * 100
                    
                    return {
                        'primary_role': primary_role[0].replace('_', ' ').title(),
                        'match_confidence': round(confidence, 2),
                        'role_scores': {
                            role.replace('_', ' ').title(): round(score * 100, 2)
                            for role, score in role_scores.items()
                        },
                        'analysis_source': 'local_models'
                    }
            
            # Fallback to keyword matching
            return self._keyword_based_prediction(resume_text, job_description)
            
        except Exception as e:
            logger.error(f"Error in role prediction: {str(e)}")
            return self._keyword_based_prediction(resume_text, job_description)

    def _keyword_based_prediction(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Fallback method using keyword matching."""
        try:
            # Combine texts for analysis
            combined_text = (resume_text + ' ' + job_description).lower()
            
            # Count role-related keywords
            role_scores = {}
            for role, keywords in self.role_categories.items():
                score = sum(1 for keyword in keywords if keyword in combined_text)
                role_scores[role] = score
            
            if role_scores:
                # Get the role with highest score
                primary_role = max(role_scores.items(), key=lambda x: x[1])
                confidence = (primary_role[1] / len(self.role_categories[primary_role[0]])) * 100
                
                return {
                    'primary_role': primary_role[0].replace('_', ' ').title(),
                    'match_confidence': round(confidence, 2),
                    'role_scores': {
                        role.replace('_', ' ').title(): round((score / len(self.role_categories[role])) * 100, 2)
                        for role, score in role_scores.items()
                    },
                    'analysis_source': 'keyword_matching'
                }
            
            return {
                'primary_role': 'Unknown',
                'match_confidence': 0,
                'role_scores': {},
                'analysis_source': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"Error in keyword-based prediction: {str(e)}")
            return {
                'primary_role': 'Unknown',
                'match_confidence': 0,
                'role_scores': {},
                'analysis_source': 'fallback'
            }

# Initialize global role predictor instance
_role_predictor = None

def get_role_predictor() -> RolePredictor:
    """Get the global role predictor instance."""
    global _role_predictor
    if _role_predictor is None:
        _role_predictor = RolePredictor()
    return _role_predictor

def predict_roles(resume_text, job_description):
    """
    Predict roles based on resume and job description.
    This is the main entry point for role prediction.
    """
    return get_role_predictor().predict_role(resume_text, job_description)

def _model_based_role_prediction(resume_text):
    """
    Fallback method using trained model for role prediction.
    """
    try:
        # Extract skills from resume
        resume_skills = extract_skills(resume_text)
        
        # Get semantic embedding
        resume_embedding = sentence_transformer.encode([resume_text])[0]
        
        # Define role-skill mappings with weights
        role_skill_mappings = {
            'Software Engineer': {
                'skills': ['programming', 'frameworks', 'tools'],
                'weights': [0.4, 0.3, 0.3]
            },
            'Data Scientist': {
                'skills': ['data_science', 'programming', 'tools'],
                'weights': [0.5, 0.3, 0.2]
            },
            'DevOps Engineer': {
                'skills': ['devops', 'cloud', 'tools'],
                'weights': [0.4, 0.4, 0.2]
            },
            'Cloud Engineer': {
                'skills': ['cloud', 'devops', 'security'],
                'weights': [0.5, 0.3, 0.2]
            },
            'Security Engineer': {
                'skills': ['security', 'devops', 'tools'],
                'weights': [0.5, 0.3, 0.2]
            },
            'Full Stack Developer': {
                'skills': ['programming', 'frameworks', 'tools'],
                'weights': [0.4, 0.4, 0.2]
            },
            'Backend Developer': {
                'skills': ['programming', 'frameworks', 'tools'],
                'weights': [0.5, 0.3, 0.2]
            },
            'Frontend Developer': {
                'skills': ['frameworks', 'tools'],
                'weights': [0.6, 0.4]
            },
            'Machine Learning Engineer': {
                'skills': ['data_science', 'programming', 'tools'],
                'weights': [0.5, 0.3, 0.2]
            },
            'Data Engineer': {
                'skills': ['data_science', 'programming', 'tools'],
                'weights': [0.4, 0.4, 0.2]
            }
        }
        
        # Calculate role scores
        role_scores = {}
        for role, mapping in role_skill_mappings.items():
            score = 0
            for skill_category, weight in zip(mapping['skills'], mapping['weights']):
                if skill_category in resume_skills:
                    # Calculate weighted score based on number of skills in category
                    category_skills = resume_skills[skill_category]
                    score += weight * (len(category_skills) / len(SKILLS_DB[skill_category]))
            role_scores[role] = score
        
        # Get top 3 roles
        top_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        return [role for role, score in top_roles if score > 0]
        
    except Exception as e:
        print(f"Error in model-based role prediction: {str(e)}")
        return _extract_roles_from_text(resume_text)

def _extract_roles_from_text(text):
    """
    Final fallback method to extract roles from text using basic pattern matching.
    """
    common_roles = [
        'Software Engineer', 'Data Scientist', 'DevOps Engineer',
        'Cloud Engineer', 'Security Engineer', 'Full Stack Developer',
        'Backend Developer', 'Frontend Developer', 'Machine Learning Engineer',
        'Data Engineer'
    ]
    
    found_roles = []
    text_lower = text.lower()
    
    for role in common_roles:
        if role.lower() in text_lower:
            found_roles.append(role)
    
    return found_roles[:3] if found_roles else ['Software Engineer', 'Full Stack Developer', 'Backend Developer'] 