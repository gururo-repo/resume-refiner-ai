import logging
from typing import Dict, Any
from sentence_transformers import SentenceTransformer, util
import torch
import joblib
import os
import re
from collections import Counter

logger = logging.getLogger(__name__)

class MLFallback:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize ML models for fallback analysis."""
        try:
            # Load models if they exist
            models_dir = os.path.join('models', 'fallback')
            os.makedirs(models_dir, exist_ok=True)
            
            # Initialize or load models
            self.strength_analyzer = self._load_or_create_model(os.path.join(models_dir, 'strength_analyzer'))
            self.weakness_analyzer = self._load_or_create_model(os.path.join(models_dir, 'weakness_analyzer'))
            self.tip_generator = self._load_or_create_model(os.path.join(models_dir, 'tip_generator'))
            
        except Exception as e:
            logger.error(f"Error initializing ML fallback models: {str(e)}")
            raise
    
    def _load_or_create_model(self, model_path: str):
        """Load existing model or create a new one."""
        try:
            if os.path.exists(model_path):
                return joblib.load(model_path)
            return None
        except Exception:
            return None
    
    def get_fallback_feedback(self, resume_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """
        Get ML-based feedback when GenAI fails.
        
        Args:
            resume_data: Parsed resume data
            job_description: Job description text
            
        Returns:
            Dictionary containing ML-based feedback
        """
        try:
            # Analyze strengths
            strengths = self._analyze_strengths(resume_data, job_description)
            
            # Analyze weaknesses
            weaknesses = self._analyze_weaknesses(resume_data, job_description)
            
            # Generate tips
            tips = self._generate_tips(resume_data, job_description)
            
            # Generate rewritten sections
            rewritten_sections = self._generate_rewritten_sections(resume_data, job_description)
            
            return {
                "strengths": strengths,
                "weaknesses": weaknesses,
                "tips": tips,
                "rewritten_sections": rewritten_sections
            }
            
        except Exception as e:
            logger.error(f"Error getting ML fallback feedback: {str(e)}")
            raise
    
    def _analyze_strengths(self, resume_data: Dict[str, Any], job_description: str) -> list:
        """Analyze resume strengths using ML."""
        strengths = []
        
        # Check for matching skills
        resume_skills = set(resume_data.get('skills', []))
        job_skills = self._extract_skills_from_text(job_description)
        matching_skills = resume_skills.intersection(job_skills)
        
        if matching_skills:
            strengths.append(f"Strong match in technical skills: {', '.join(matching_skills)}")
        
        # Check for relevant experience
        if 'experience' in resume_data['sections']:
            exp_text = resume_data['sections']['experience']
            similarity = self._calculate_similarity(exp_text, job_description)
            if similarity > 0.7:
                strengths.append("Highly relevant work experience")
            elif similarity > 0.5:
                strengths.append("Moderately relevant work experience")
        
        # Check for education match
        if 'education' in resume_data['sections']:
            edu_text = resume_data['sections']['education']
            if any(keyword in edu_text.lower() for keyword in ['bachelor', 'master', 'phd']):
                strengths.append("Strong educational background")
        
        return strengths
    
    def _analyze_weaknesses(self, resume_data: Dict[str, Any], job_description: str) -> list:
        """Analyze resume weaknesses using ML."""
        weaknesses = []
        
        # Check for missing skills
        resume_skills = set(resume_data.get('skills', []))
        job_skills = self._extract_skills_from_text(job_description)
        missing_skills = job_skills - resume_skills
        
        if missing_skills:
            weaknesses.append(f"Missing key skills: {', '.join(missing_skills)}")
        
        # Check for experience gaps
        if 'experience' in resume_data['sections']:
            exp_text = resume_data['sections']['experience']
            similarity = self._calculate_similarity(exp_text, job_description)
            if similarity < 0.3:
                weaknesses.append("Limited relevant work experience")
        
        # Check for education gaps
        if 'education' in resume_data['sections']:
            edu_text = resume_data['sections']['education']
            if not any(keyword in edu_text.lower() for keyword in ['bachelor', 'master', 'phd']):
                weaknesses.append("May need additional education or certifications")
        
        return weaknesses
    
    def _generate_tips(self, resume_data: Dict[str, Any], job_description: str) -> list:
        """Generate improvement tips using ML."""
        tips = []
        
        # Check for quantifiable achievements
        if 'experience' in resume_data['sections']:
            exp_text = resume_data['sections']['experience']
            if not re.search(r'\d+%|\d+x|\$\d+', exp_text):
                tips.append("Add quantifiable achievements to experience section")
        
        # Check for skills organization
        if 'skills' in resume_data:
            if len(resume_data['skills']) < 5:
                tips.append("Expand skills section with more relevant technical skills")
        
        # Check for summary section
        if 'summary' not in resume_data['sections']:
            tips.append("Add a professional summary section")
        
        # Check for formatting
        if len(resume_data['raw_text'].split('\n')) < 10:
            tips.append("Improve resume formatting and structure")
        
        return tips
    
    def _generate_rewritten_sections(self, resume_data: Dict[str, Any], job_description: str) -> Dict[str, str]:
        """Generate improved versions of key sections."""
        rewritten = {
            "summary": "",
            "experience": "",
            "skills": ""
        }
        
        # Generate improved summary
        if 'summary' in resume_data['sections']:
            summary = resume_data['sections']['summary']
            rewritten['summary'] = self._improve_summary(summary, job_description)
        
        # Generate improved experience
        if 'experience' in resume_data['sections']:
            experience = resume_data['sections']['experience']
            rewritten['experience'] = self._improve_experience(experience, job_description)
        
        # Generate improved skills
        if 'skills' in resume_data:
            skills = resume_data['skills']
            rewritten['skills'] = self._improve_skills(skills, job_description)
        
        return rewritten
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            embedding1 = self.model.encode(text1, convert_to_tensor=True)
            embedding2 = self.model.encode(text2, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embedding1, embedding2)
            return float(similarity[0][0])
        except Exception:
            return 0.0
    
    def _extract_skills_from_text(self, text: str) -> set:
        """Extract skills from text."""
        # Common technical skills
        technical_skills = {
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php',
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
            'django', 'flask', 'spring', 'sql', 'nosql', 'mongodb', 'postgresql',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'agile',
            'scrum', 'machine learning', 'ai', 'data science', 'big data',
            'devops', 'ci/cd', 'testing', 'security', 'networking'
        }
        
        text = text.lower()
        return {skill for skill in technical_skills if skill in text}
    
    def _improve_summary(self, summary: str, job_description: str) -> str:
        """Generate improved summary section."""
        # Basic improvement logic
        improved = summary.strip()
        if not improved.endswith('.'):
            improved += '.'
        return improved
    
    def _improve_experience(self, experience: str, job_description: str) -> str:
        """Generate improved experience section."""
        # Basic improvement logic
        improved = experience.strip()
        if not improved.endswith('.'):
            improved += '.'
        return improved
    
    def _improve_skills(self, skills: list, job_description: str) -> str:
        """Generate improved skills section."""
        # Basic improvement logic
        return ', '.join(skills)

# Initialize global fallback instance
ml_fallback = MLFallback()

def get_ml_fallback(resume_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
    """
    Get ML-based feedback when GenAI fails.
    
    Args:
        resume_data: Parsed resume data
        job_description: Job description text
        
    Returns:
        Dictionary containing ML-based feedback
    """
    return ml_fallback.get_fallback_feedback(resume_data, job_description) 