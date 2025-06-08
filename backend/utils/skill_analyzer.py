import logging
from typing import Dict, Any, List, Set
from sentence_transformers import SentenceTransformer, util
import torch
import re
from collections import Counter

logger = logging.getLogger(__name__)

class SkillAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define skill categories and their associated keywords
        self.skill_categories = {
            'programming_languages': {
                'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php',
                'swift', 'kotlin', 'go', 'rust', 'typescript'
            },
            'web_technologies': {
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
                'django', 'flask', 'spring', 'asp.net', 'jquery'
            },
            'databases': {
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra',
                'oracle', 'sql server', 'dynamodb', 'firebase'
            },
            'cloud_platforms': {
                'aws', 'azure', 'gcp', 'heroku', 'digitalocean', 'linode',
                'cloudflare', 'alibaba cloud'
            },
            'devops_tools': {
                'docker', 'kubernetes', 'jenkins', 'git', 'ansible', 'terraform',
                'prometheus', 'grafana', 'elk stack'
            },
            'machine_learning': {
                'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'pandas',
                'numpy', 'matplotlib', 'seaborn', 'opencv'
            },
            'soft_skills': {
                'communication', 'leadership', 'teamwork', 'problem-solving',
                'time management', 'adaptability', 'creativity', 'critical thinking'
            }
        }
    
    def analyze_missing_skills(self, resume_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """
        Analyze missing skills by comparing resume skills with job requirements.
        
        Args:
            resume_data: Parsed resume data
            job_description: Job description text
            
        Returns:
            Dictionary containing missing skills analysis
        """
        try:
            # Extract skills from resume and job description
            resume_skills = self._extract_skills_from_resume(resume_data)
            job_skills = self._extract_skills_from_text(job_description)
            
            # Find missing skills
            missing_skills = self._find_missing_skills(resume_skills, job_skills)
            
            # Categorize missing skills
            categorized_skills = self._categorize_skills(missing_skills)
            
            # Calculate skill gap score
            skill_gap_score = self._calculate_skill_gap_score(resume_skills, job_skills)
            
            return {
                'missing_skills': categorized_skills,
                'skill_gap_score': skill_gap_score,
                'recommendations': self._generate_skill_recommendations(categorized_skills)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing missing skills: {str(e)}")
            raise
    
    def _extract_skills_from_resume(self, resume_data: Dict[str, Any]) -> Set[str]:
        """Extract skills from resume data."""
        skills = set()
        
        # Add skills from skills section
        if 'skills' in resume_data:
            skills.update(resume_data['skills'])
        
        # Extract skills from experience section
        if 'experience' in resume_data['sections']:
            exp_skills = self._extract_skills_from_text(resume_data['sections']['experience'])
            skills.update(exp_skills)
        
        return skills
    
    def _extract_skills_from_text(self, text: str) -> Set[str]:
        """Extract skills from text using predefined skill categories."""
        text = text.lower()
        found_skills = set()
        
        # Check each category for skills
        for category, skills in self.skill_categories.items():
            for skill in skills:
                if skill.lower() in text:
                    found_skills.add(skill.lower())
        
        return found_skills
    
    def _find_missing_skills(self, resume_skills: Set[str], job_skills: Set[str]) -> Set[str]:
        """Find skills that are in job description but not in resume."""
        return job_skills - resume_skills
    
    def _categorize_skills(self, skills: Set[str]) -> Dict[str, List[str]]:
        """Categorize skills into predefined categories."""
        categorized = {}
        
        for category, category_skills in self.skill_categories.items():
            category_matches = [skill for skill in skills if skill in category_skills]
            if category_matches:
                categorized[category] = category_matches
        
        return categorized
    
    def _calculate_skill_gap_score(self, resume_skills: Set[str], job_skills: Set[str]) -> float:
        """Calculate skill gap score (0-100)."""
        if not job_skills:
            return 0.0
        
        matching_skills = resume_skills.intersection(job_skills)
        return round((len(matching_skills) / len(job_skills)) * 100, 2)
    
    def _generate_skill_recommendations(self, categorized_skills: Dict[str, List[str]]) -> List[str]:
        """Generate recommendations for acquiring missing skills."""
        recommendations = []
        
        for category, skills in categorized_skills.items():
            if skills:
                category_name = category.replace('_', ' ').title()
                recommendations.append(
                    f"Consider developing your {category_name} skills, particularly in: {', '.join(skills)}"
                )
        
        return recommendations

# Initialize global analyzer instance
skill_analyzer = SkillAnalyzer()

def analyze_missing_skills(resume_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
    """
    Analyze missing skills in the resume compared to job requirements.
    
    Args:
        resume_data: Parsed resume data
        job_description: Job description text
        
    Returns:
        Dictionary containing missing skills analysis
    """
    return skill_analyzer.analyze_missing_skills(resume_data, job_description) 