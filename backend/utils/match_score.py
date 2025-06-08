import json
import numpy as np
import logging
from .genai_suggester import ResumeImprover
from .model_loader import get_models, get_match_model, get_feature_scaler, get_model_features, get_sentence_transformer, get_model_loader
from .skill_matcher import extract_skills, SKILLS_DB
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from typing import Dict, Any, Tuple
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchScoreCalculator:
    def __init__(self):
        """Initialize the MatchScoreCalculator."""
        self.model_loader = get_model_loader()
        logger.info("Successfully initialized MatchScoreCalculator")
    
    def extract_experience_level(self, text: str) -> str:
        """Extract required experience level from text."""
        try:
            # Common experience level patterns
            patterns = {
                'entry': r'entry[- ]level|junior|0-2\s*years?|1-3\s*years?',
                'mid': r'mid[- ]level|intermediate|3-5\s*years?|4-6\s*years?',
                'senior': r'senior|lead|5-7\s*years?|6-8\s*years?',
                'expert': r'expert|principal|architect|7\+\s*years?|8\+\s*years?'
            }
            
            text_lower = text.lower()
            for level, pattern in patterns.items():
                if re.search(pattern, text_lower):
                    return level
            
            return 'Not specified'
            
        except Exception as e:
            logger.error(f"Error extracting experience level: {str(e)}")
            return 'Not specified'
    
    def extract_education_requirements(self, text: str) -> str:
        """Extract education requirements from text."""
        try:
            # Common education patterns
            patterns = {
                'high_school': r'high school|GED|diploma',
                'associate': r'associate|AA|AS|AAS',
                'bachelor': r'bachelor|BS|BA|BSc|BEng',
                'master': r'master|MS|MA|MSc|MEng',
                'phd': r'phd|doctorate|doctoral'
            }
            
            text_lower = text.lower()
            for level, pattern in patterns.items():
                if re.search(pattern, text_lower):
                    return level
            
            return 'Not specified'
            
        except Exception as e:
            logger.error(f"Error extracting education requirements: {str(e)}")
            return 'Not specified'
    
    def calculate_match_score(self, resume_data: Dict[str, Any], job_description: str) -> float:
        """Calculate overall match score between resume and job description."""
        try:
            # Get match score from model loader
            match_score = self.model_loader.predict_job_match(resume_data, job_description)
            return match_score
            
        except Exception as e:
            logger.error(f"Error calculating match score: {str(e)}")
            return 0.0
    
    def analyze_match_components(self, resume_data: Dict[str, Any], job_description: str) -> Dict[str, float]:
        """Analyze different components of the match."""
        try:
            # Extract skills from both
            resume_skills = set(resume_data.get('skills', []))
            job_skills = set(self.model_loader._extract_skills_from_text(job_description))
            
            # Calculate skill match
            skill_match = len(resume_skills.intersection(job_skills)) / max(1, len(job_skills))
            
            # Calculate experience match
            experience_match = self._calculate_experience_match(resume_data, job_description)
            
            # Calculate education match
            education_match = self._calculate_education_match(resume_data, job_description)
            
            return {
                'skill_match': skill_match,
                'experience_match': experience_match,
                'education_match': education_match
            }
            
        except Exception as e:
            logger.error(f"Error analyzing match components: {str(e)}")
            return {
                'skill_match': 0.0,
                'experience_match': 0.0,
                'education_match': 0.0
            }
    
    def _calculate_experience_match(self, resume_data: Dict[str, Any], job_description: str) -> float:
        """Calculate experience match score."""
        try:
            # Extract experience from resume
            experience_text = resume_data.get('sections', {}).get('experience', '')
            
            # Get experience level from job description
            required_level = self.extract_experience_level(job_description)
            
            # Simple matching based on keywords
            if required_level == 'entry':
                return 1.0 if 'entry' in experience_text.lower() or 'junior' in experience_text.lower() else 0.5
            elif required_level == 'mid':
                return 1.0 if 'mid' in experience_text.lower() or 'intermediate' in experience_text.lower() else 0.5
            elif required_level == 'senior':
                return 1.0 if 'senior' in experience_text.lower() or 'lead' in experience_text.lower() else 0.5
            elif required_level == 'expert':
                return 1.0 if 'expert' in experience_text.lower() or 'principal' in experience_text.lower() else 0.5
            
            return 0.5  # Default score if no clear match
            
        except Exception as e:
            logger.error(f"Error calculating experience match: {str(e)}")
            return 0.0
    
    def _calculate_education_match(self, resume_data: Dict[str, Any], job_description: str) -> float:
        """Calculate education match score."""
        try:
            # Extract education from resume
            education_text = resume_data.get('sections', {}).get('education', '')
            
            # Get education requirement from job description
            required_education = self.extract_education_requirements(job_description)
            
            # Simple matching based on keywords
            if required_education == 'high_school':
                return 1.0
            elif required_education == 'associate':
                return 1.0 if 'associate' in education_text.lower() else 0.5
            elif required_education == 'bachelor':
                return 1.0 if 'bachelor' in education_text.lower() else 0.5
            elif required_education == 'master':
                return 1.0 if 'master' in education_text.lower() else 0.5
            elif required_education == 'phd':
                return 1.0 if 'phd' in education_text.lower() or 'doctorate' in education_text.lower() else 0.5
            
            return 0.5  # Default score if no clear match
            
        except Exception as e:
            logger.error(f"Error calculating education match: {str(e)}")
            return 0.0

# Create a singleton instance
match_score_calculator = MatchScoreCalculator()

def get_match_score(resume_skills, job_skills, resume_text, job_description):
    """
    Get match score between resume and job description.
    This is the main entry point for match score calculation.
    """
    return match_score_calculator.calculate_match_score(resume_skills, job_skills, resume_text, job_description)

def _model_based_analysis(resume_text, job_description):
    """
    Fallback method using trained model for match analysis.
    """
    try:
        # Extract skills from both texts
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)
        
        # Calculate skill match metrics
        resume_skill_set = set()
        job_skill_set = set()
        
        for category, skills in resume_skills.items():
            resume_skill_set.update(skills)
        
        for category, skills in job_skills.items():
            job_skill_set.update(skills)
        
        # Calculate Jaccard similarity
        if len(resume_skill_set) == 0 or len(job_skill_set) == 0:
            jaccard_similarity = 0.0
        else:
            intersection = resume_skill_set.intersection(job_skill_set)
            union = resume_skill_set.union(job_skill_set)
            jaccard_similarity = len(intersection) / len(union)
        
        # Calculate common skills count
        common_skills = resume_skill_set.intersection(job_skill_set)
        common_skills_count = len(common_skills)
        
        # Get embeddings for semantic similarity
        resume_embedding = sentence_transformer.encode([resume_text])[0]
        job_embedding = sentence_transformer.encode([job_description])[0]
        skill_vector_similarity = np.dot(resume_embedding, job_embedding) / (
            np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
        )
        
        # Calculate additional features
        skill_count_diff = abs(len(resume_skill_set) - len(job_skill_set))
        skill_count_ratio = common_skills_count / max(1, len(job_skill_set))
        skills_density = len(resume_skill_set) / max(1, len(resume_text.split()))
        
        # Prepare features for model prediction
        features = np.array([[
            jaccard_similarity,
            common_skills_count,
            skill_vector_similarity,
            skill_count_diff,
            skill_count_ratio,
            skills_density
        ]])
        
        # Scale features
        features_scaled = feature_scaler.transform(features)
        
        # Get match score prediction
        match_score = match_model.predict(features_scaled)[0]
        match_percentage = min(100, max(0, match_score * 100))
        
        # Calculate ATS score
        ats_score = min(100, max(0, (
            jaccard_similarity * 0.4 +
            skill_vector_similarity * 0.3 +
            skill_count_ratio * 0.3
        ) * 100))
        
        # Generate strengths and weaknesses
        strengths = [f"Strong skill: {skill}" for skill in list(common_skills)[:5]]
        weaknesses = [f"Missing skill: {skill}" for skill in list(job_skill_set - resume_skill_set)[:5]]
        
        # Generate recommendations
        recommendations = []
        if weaknesses:
            recommendations.append(f"Add missing skills: {', '.join(list(job_skill_set - resume_skill_set)[:3])}")
        if skills_density < 0.1:
            recommendations.append("Increase skills density in resume")
        if len(resume_text.split()) < 200:
            recommendations.append("Add more detailed content to resume")
        
        return {
            'match_score': round(match_percentage, 2),
            'ats_score': round(ats_score, 2),
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations
        }
        
    except Exception as e:
        print(f"Error in model-based analysis: {str(e)}")
        return _basic_match_analysis(resume_text, job_description)

def _basic_match_analysis(resume_text, job_description):
    """
    Final fallback method for basic match analysis.
    """
    return {
        'match_score': 50,
        'ats_score': 50,
        'strengths': ['Basic resume structure present'],
        'weaknesses': ['Unable to perform detailed analysis'],
        'recommendations': ['Please try again or contact support']
    } 