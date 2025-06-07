import os
import json
import numpy as np
import logging
from .genai_suggester import ResumeImprover
from .model_loader import get_models, get_match_model, get_feature_scaler, get_model_features, get_sentence_transformer
from .skill_matcher import extract_skills, SKILLS_DB
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RolePredictor:
    def __init__(self):
        try:
            self.models = get_models()
            self.match_model = get_match_model()
            self.feature_scaler = get_feature_scaler()
            self.model_features = get_model_features()
            self.sentence_transformer = get_sentence_transformer()
            logger.info("Successfully initialized RolePredictor")
        except Exception as e:
            logger.error(f"Failed to initialize RolePredictor: {str(e)}")
            raise

    def predict_roles(self, resume_text, job_description):
        """
        Predict roles based on resume and job description.
        First attempts LLM-based analysis, falls back to model-based analysis if LLM fails.
        """
        try:
            # First attempt: Use LLM-based analysis
            suggester = ResumeImprover()
            analysis = suggester.analyze_roles(resume_text, job_description)
            
            # Extract roles from LLM analysis
            predicted_roles = analysis.get('predicted_roles', [])
            confidence_scores = analysis.get('confidence_scores', [])
            
            logger.info("Successfully completed LLM-based role prediction")
            return {
                'predicted_roles': predicted_roles,
                'confidence_scores': confidence_scores,
                'analysis_type': 'llm'
            }
            
        except Exception as e:
            logger.warning(f"LLM analysis failed, falling back to model-based analysis: {str(e)}")
            return self._model_based_role_prediction(resume_text, job_description)

    def _model_based_role_prediction(self, resume_text, job_description):
        """
        Fallback method using trained model for role prediction.
        Only used when LLM analysis fails.
        """
        try:
            # Get text embeddings
            resume_embedding = self.sentence_transformer.encode(resume_text)
            job_embedding = self.sentence_transformer.encode(job_description)
            
            # Combine embeddings for role prediction
            combined_embedding = (resume_embedding + job_embedding) / 2
            
            # Compare with known role embeddings
            roles = []
            for role, embedding in self.model_features.items():
                similarity = cosine_similarity([combined_embedding], [embedding])[0][0]
                if similarity > 0.5:  # Threshold for role prediction
                    roles.append({
                        'role': role,
                        'confidence': float(similarity)
                    })
            
            # Sort by confidence and get top 3
            roles.sort(key=lambda x: x['confidence'], reverse=True)
            top_roles = roles[:3]
            
            logger.info("Successfully completed model-based role prediction")
            return {
                'predicted_roles': [role['role'] for role in top_roles],
                'confidence_scores': [role['confidence'] for role in top_roles],
                'analysis_type': 'model'
            }
            
        except Exception as e:
            logger.error(f"Model-based role prediction failed: {str(e)}")
            return {
                'predicted_roles': [],
                'confidence_scores': [],
                'analysis_type': 'fallback'
            }

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