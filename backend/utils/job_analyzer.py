import logging
from typing import Dict, Any, List
from .model_loader import get_model_loader
from .skill_matcher import SkillMatcher
from .match_score import MatchScoreCalculator
from .role_predictor import RolePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_job_description(job_description: str) -> Dict[str, Any]:
    """
    Analyze a job description to extract key information and requirements.
    
    Args:
        job_description (str): The job description text to analyze
        
    Returns:
        Dict[str, Any]: Analysis results including:
            - required_skills: List of required skills
            - preferred_skills: List of preferred skills
            - experience_level: Required experience level
            - education_requirements: Required education
            - role_category: Predicted role category
            - role_confidence: Confidence score for role prediction
    """
    try:
        # Initialize components
        model_loader = get_model_loader()
        skill_matcher = SkillMatcher()
        match_calculator = MatchScoreCalculator()
        role_predictor = RolePredictor()
        
        # Extract skills
        required_skills = skill_matcher.extract_required_skills(job_description)
        preferred_skills = skill_matcher.extract_preferred_skills(job_description)
        
        # Analyze experience and education requirements
        experience_level = match_calculator.extract_experience_level(job_description)
        education_requirements = match_calculator.extract_education_requirements(job_description)
        
        # Predict role category
        role_category, role_confidence = role_predictor.predict_role(job_description)
        
        return {
            'required_skills': required_skills,
            'preferred_skills': preferred_skills,
            'experience_level': experience_level,
            'education_requirements': education_requirements,
            'role_category': role_category,
            'role_confidence': role_confidence
        }
        
    except Exception as e:
        logger.error(f"Error analyzing job description: {str(e)}")
        return {
            'required_skills': [],
            'preferred_skills': [],
            'experience_level': 'Not specified',
            'education_requirements': 'Not specified',
            'role_category': 'Unknown',
            'role_confidence': 0.0
        } 