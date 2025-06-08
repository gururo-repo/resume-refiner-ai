import logging
from typing import Dict, Any, List
import joblib
import os
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class RoleRecommender:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.initialize_model()
        
        # Define role categories and their associated keywords
        self.role_categories = {
            'software_development': {
                'keywords': ['software', 'developer', 'programming', 'coding', 'application'],
                'sub_roles': ['frontend', 'backend', 'fullstack', 'mobile', 'game']
            },
            'data_science': {
                'keywords': ['data', 'analytics', 'machine learning', 'ai', 'statistics'],
                'sub_roles': ['data scientist', 'data analyst', 'ml engineer', 'data engineer']
            },
            'devops': {
                'keywords': ['devops', 'cloud', 'infrastructure', 'automation', 'ci/cd'],
                'sub_roles': ['devops engineer', 'cloud engineer', 'site reliability']
            },
            'security': {
                'keywords': ['security', 'cybersecurity', 'information security', 'network security'],
                'sub_roles': ['security engineer', 'security analyst', 'penetration tester']
            },
            'product_management': {
                'keywords': ['product', 'management', 'agile', 'scrum', 'project'],
                'sub_roles': ['product manager', 'project manager', 'product owner']
            }
        }
    
    def initialize_model(self):
        """Initialize the role recommendation model."""
        try:
            model_path = os.path.join('models', 'role_recommender')
            if os.path.exists(model_path):
                logger.info("Loading fine-tuned role recommender model...")
                self.model = joblib.load(model_path)
            else:
                logger.info("Using base Sentence-BERT model for role recommendations...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error initializing role recommender model: {str(e)}")
            raise
    
    def get_recommendations(self, resume_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get role recommendations based on resume content.
        
        Args:
            resume_data: Parsed resume data
            
        Returns:
            List of recommended roles with confidence scores
        """
        try:
            # Prepare resume text
            resume_text = self._prepare_resume_text(resume_data)
            
            # Get role scores
            role_scores = self._calculate_role_scores(resume_text)
            
            # Get top recommendations
            recommendations = self._get_top_recommendations(role_scores)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting role recommendations: {str(e)}")
            raise
    
    def _prepare_resume_text(self, resume_data: Dict[str, Any]) -> str:
        """Prepare resume text for role analysis."""
        sections = []
        
        # Add skills section
        if 'skills' in resume_data:
            sections.append(' '.join(resume_data['skills']))
        
        # Add experience section
        if 'experience' in resume_data['sections']:
            sections.append(resume_data['sections']['experience'])
        
        return ' '.join(sections)
    
    def _calculate_role_scores(self, text: str) -> Dict[str, float]:
        """Calculate scores for each role category."""
        scores = {}
        text = text.lower()
        
        for role, info in self.role_categories.items():
            # Calculate keyword match score
            keyword_matches = sum(1 for keyword in info['keywords'] if keyword in text)
            keyword_score = keyword_matches / len(info['keywords'])
            
            # Calculate sub-role match score
            sub_role_matches = sum(1 for sub_role in info['sub_roles'] if sub_role in text)
            sub_role_score = sub_role_matches / len(info['sub_roles'])
            
            # Combine scores
            scores[role] = (keyword_score * 0.6 + sub_role_score * 0.4)
        
        return scores
    
    def _get_top_recommendations(self, role_scores: Dict[str, float], top_n: int = 3) -> List[Dict[str, Any]]:
        """Get top N role recommendations with confidence scores."""
        # Sort roles by score
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for role, score in sorted_roles[:top_n]:
            if score > 0:  # Only include roles with positive scores
                recommendations.append({
                    'role': role.replace('_', ' ').title(),
                    'confidence': round(score * 100, 2),
                    'sub_roles': self.role_categories[role]['sub_roles']
                })
        
        return recommendations

# Initialize global recommender instance
role_recommender = RoleRecommender()

def get_role_recommendations(resume_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get role recommendations for the resume.
    
    Args:
        resume_data: Parsed resume data
        
    Returns:
        List of recommended roles with confidence scores
    """
    return role_recommender.get_recommendations(resume_data) 