import logging
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer, util
import torch
import re
from collections import Counter
from .model_loader import get_model_loader

logger = logging.getLogger(__name__)

class JobMatcher:
    def __init__(self):
        self.model_loader = get_model_loader()
        self.skill_weights = {
            'technical_skills': 0.4,
            'soft_skills': 0.2,
            'experience': 0.3,
            'education': 0.1
        }
    
    def calculate_match_score(self, resume_data: Dict[str, Any], job_description: str) -> float:
        """
        Calculate overall job match score using the trained model.
        
        Args:
            resume_data: Parsed resume data
            job_description: Job description text
            
        Returns:
            Job match score (0-100)
        """
        try:
            # Get prediction from trained model
            match_score = self.model_loader.predict_job_match(resume_data, job_description)
            
            # Convert to percentage
            return round(match_score * 100, 2)
            
        except Exception as e:
            logger.error(f"Error calculating job match score: {str(e)}")
            raise
    
    def _calculate_skill_match(self, resume_data: Dict[str, Any], job_description: str) -> float:
        """Calculate skill match score."""
        try:
            # Extract skills from job description
            job_skills = self._extract_skills_from_text(job_description)
            
            # Get resume skills
            resume_skills = set(resume_data.get('skills', []))
            
            # Calculate skill match
            if not job_skills:
                return 0.0
            
            matching_skills = resume_skills.intersection(job_skills)
            return len(matching_skills) / len(job_skills)
            
        except Exception as e:
            logger.error(f"Error calculating skill match: {str(e)}")
            return 0.0
    
    def _calculate_experience_match(self, resume_data: Dict[str, Any], job_description: str) -> float:
        """Calculate experience match score using semantic similarity."""
        try:
            if 'experience' not in resume_data['sections']:
                return 0.0
            
            resume_exp = resume_data['sections']['experience']
            
            # Encode texts
            resume_embedding = self.model_loader.sentence_model.encode(resume_exp, convert_to_tensor=True)
            job_embedding = self.model_loader.sentence_model.encode(job_description, convert_to_tensor=True)
            
            # Calculate similarity
            similarity = util.pytorch_cos_sim(resume_embedding, job_embedding)
            return float(similarity[0][0])
            
        except Exception as e:
            logger.error(f"Error calculating experience match: {str(e)}")
            return 0.0
    
    def _calculate_education_match(self, resume_data: Dict[str, Any], job_description: str) -> float:
        """Calculate education match score."""
        try:
            if 'education' not in resume_data['sections']:
                return 0.0
            
            resume_edu = resume_data['sections']['education'].lower()
            job_desc = job_description.lower()
            
            # Check for common education requirements
            education_keywords = {
                'bachelor': 1.0,
                'master': 1.0,
                'phd': 1.0,
                'degree': 0.8,
                'diploma': 0.6,
                'certification': 0.4
            }
            
            max_score = 0.0
            for keyword, score in education_keywords.items():
                if keyword in job_desc and keyword in resume_edu:
                    max_score = max(max_score, score)
            
            return max_score
            
        except Exception as e:
            logger.error(f"Error calculating education match: {str(e)}")
            return 0.0
    
    def _extract_skills_from_text(self, text: str) -> set:
        """Extract skills from text using common skill patterns."""
        # Common technical skills
        technical_skills = {
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php',
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
            'django', 'flask', 'spring', 'sql', 'nosql', 'mongodb', 'postgresql',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'agile',
            'scrum', 'machine learning', 'ai', 'data science', 'big data',
            'devops', 'ci/cd', 'testing', 'security', 'networking'
        }
        
        # Convert text to lowercase for matching
        text = text.lower()
        
        # Find matching skills
        found_skills = set()
        for skill in technical_skills:
            if skill in text:
                found_skills.add(skill)
        
        return found_skills

# Initialize global matcher instance
job_matcher = JobMatcher()

def calculate_job_match(resume_data: Dict[str, Any], job_description: str) -> float:
    """
    Calculate job match score.
    
    Args:
        resume_data: Parsed resume data
        job_description: Job description text
        
    Returns:
        Job match score (0-100)
    """
    return job_matcher.calculate_match_score(resume_data, job_description) 