import os
import json
import numpy as np
import logging
from .genai_suggester import ResumeImprover
from .model_loader import get_models, get_match_model, get_feature_scaler, get_model_features, get_sentence_transformer, get_model_loader
from .skill_matcher import extract_skills, SKILLS_DB
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, Tuple, List
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RolePredictor:
    def __init__(self):
        """Initialize the RolePredictor."""
        self.genai_model = None
        self._initialize_genai()
        self.role_categories = {
            'software_engineering': [
                'software engineer', 'developer', 'programmer', 'full stack', 'frontend', 'backend',
                'web developer', 'mobile developer', 'devops', 'sre', 'qa engineer'
            ],
            'data_science': [
                'data scientist', 'data analyst', 'machine learning', 'ai engineer', 'ml engineer',
                'data engineer', 'business intelligence', 'bi analyst'
            ],
            'product_management': [
                'product manager', 'product owner', 'project manager', 'technical product manager',
                'product analyst', 'product marketing'
            ],
            'design': [
                'ui designer', 'ux designer', 'graphic designer', 'interaction designer',
                'visual designer', 'product designer'
            ],
            'marketing': [
                'marketing manager', 'digital marketing', 'content marketing', 'social media',
                'growth marketing', 'seo specialist'
            ],
            'sales': [
                'sales representative', 'account executive', 'business development',
                'sales manager', 'account manager'
            ]
        }
        logger.info("Successfully initialized RolePredictor")

    def _initialize_genai(self):
        """Initialize Google Generative AI."""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.warning("Google API key not found. GenAI features will be disabled.")
                return
            
            genai.configure(api_key=api_key)
            
            # List available models and their capabilities
            models = genai.list_models()
            available_models = [model.name for model in models]
            logger.info(f"Available GenAI models: {available_models}")
            
            # Try to use gemini-1.5-pro first
            model = "models/gemini-1.5-pro"
            url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
            
            if model in available_models:
                self.genai_model = genai.GenerativeModel(
                    model_name=model,
                    generation_config={
                        'temperature': 0.7,
                        'top_p': 0.8,
                        'top_k': 40,
                        'max_output_tokens': 2048,
                    }
                )
                logger.info(f"Successfully initialized GenAI model: {model}")
                return
            
            # Fallback to gemini-1.0-pro
            model = "models/gemini-1.0-pro"
            url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
            
            if model in available_models:
                self.genai_model = genai.GenerativeModel(
                    model_name=model,
                    generation_config={
                        'temperature': 0.7,
                        'top_p': 0.8,
                        'top_k': 40,
                        'max_output_tokens': 2048,
                    }
                )
                logger.info(f"Successfully initialized GenAI model: {model}")
                return
            
            logger.warning("No suitable GenAI model found. GenAI features will be disabled.")
            self.genai_model = None
            
        except Exception as e:
            logger.error(f"Error initializing GenAI: {str(e)}")
            self.genai_model = None

    def predict_role(self, job_description: str) -> Tuple[str, float]:
        """
        Predict the role category from job description.
        Returns (role_category, confidence_score)
        """
        try:
            # Try GenAI first
            if self.genai_model:
                try:
                    role, confidence = self._get_genai_role_prediction(job_description)
                    if role and confidence:
                        return role, confidence
                except Exception as e:
                    logger.warning(f"GenAI prediction failed, falling back to keyword matching: {str(e)}")
            
            # Fallback to keyword matching
            return self._keyword_based_role_prediction(job_description)
            
        except Exception as e:
            logger.error(f"Error predicting role: {str(e)}")
            return 'unknown', 0.0

    def _get_genai_role_prediction(self, job_description: str) -> Tuple[str, float]:
        """Get role prediction using GenAI."""
        try:
            if not self.genai_model:
                logger.warning("GenAI model not initialized")
                return None, None

            prompt = self._prepare_role_prompt(job_description)
            try:
                # Configure generation parameters for better results
                generation_config = {
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 2048,
                }
                
                response = self.genai_model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                response_text = response.text.strip().lower()
                
                # Extract role and confidence
                if 'role:' in response_text and 'confidence:' in response_text:
                    role_part = response_text.split('role:')[1].split('confidence:')[0].strip()
                    confidence_part = response_text.split('confidence:')[1].strip()
                    
                    # Get role
                    role = role_part.split()[0]  # Take first word as role
                    
                    # Get confidence
                    try:
                        confidence = float(confidence_part.replace('%', '')) / 100
                    except ValueError:
                        confidence = 0.5  # Default confidence if parsing fails
                    
                    return role, confidence
                else:
                    logger.warning("Could not parse GenAI response format")
                    return None, None
                    
            except Exception as e:
                logger.warning(f"Error generating content with GenAI: {str(e)}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error getting GenAI role prediction: {str(e)}")
            return None, None

    def _prepare_role_prompt(self, job_description: str) -> str:
        """Prepare prompt for GenAI role prediction."""
        return f"""Analyze this job description and predict the role category.
        Job Description:
        {job_description}
        
        Respond in this format:
        Role: [role_category]
        Confidence: [confidence_score as percentage]
        
        Role categories should be one of: {', '.join(self.role_categories.keys())}
        Confidence should be a number between 0 and 100."""

    def _keyword_based_role_prediction(self, job_description: str) -> Tuple[str, float]:
        """Predict role based on keyword matching."""
        try:
            job_desc_lower = job_description.lower()
            max_matches = 0
            best_role = 'unknown'
            
            # Count keyword matches for each role
            role_matches = {}
            for role, keywords in self.role_categories.items():
                matches = sum(1 for keyword in keywords if keyword in job_desc_lower)
                role_matches[role] = matches
                if matches > max_matches:
                    max_matches = matches
                    best_role = role
            
            # Calculate confidence based on matches
            total_keywords = sum(len(keywords) for keywords in self.role_categories.values())
            confidence = max_matches / total_keywords if total_keywords > 0 else 0.0
            
            return best_role, confidence
            
        except Exception as e:
            logger.error(f"Error in keyword-based role prediction: {str(e)}")
            return 'unknown', 0.0

# Create a singleton instance
role_predictor = RolePredictor()

def predict_roles(resume_text, job_description):
    """
    Predict roles based on resume and job description.
    This is the main entry point for role prediction.
    """
    return role_predictor.predict_roles(resume_text, job_description)

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