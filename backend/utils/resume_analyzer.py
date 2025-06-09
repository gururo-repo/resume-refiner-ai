import logging
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .model_loader import get_model_loader
from .role_predictor import get_role_predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeAnalyzer:
    """Analyzes resumes and provides comprehensive feedback."""
    
    def __init__(self):
        """Initialize the resume analyzer."""
        self.model_loader = get_model_loader()
        self.role_predictor = get_role_predictor()

    def analyze_resume(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Analyze resume against job description.
        
        Args:
            resume_text: The resume text to analyze
            job_description: The job description text
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # First try Groq analysis
            groq_analysis = self.model_loader.try_groq_analysis(resume_text, job_description)
            if groq_analysis:
                logger.info("Using Groq analysis results")
                return groq_analysis

            # If Groq fails, fall back to local models
            logger.info("Falling back to local model analysis")
            
            # Get embeddings for both texts
            texts = [resume_text, job_description]
            embeddings = self.model_loader.get_embeddings(texts)
            
            if embeddings is not None:
                # Calculate similarity
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                match_score = float(similarity * 100)
            else:
                # Fallback to basic matching
                match_score = self._calculate_basic_match(resume_text, job_description)

            # Get role prediction
            role_analysis = self.role_predictor.predict_role(resume_text, job_description)
            
            # Analyze skills
            skills_analysis = self._analyze_skills(resume_text, job_description)
            
            # Analyze format
            format_analysis = self._analyze_format(resume_text)
            
            # Generate improvements
            improvements = self._generate_improvements(
                resume_text, job_description,
                skills_analysis, format_analysis
            )

            return {
                'match_score': round(match_score, 2),
                'role_analysis': role_analysis,
                'skills_analysis': skills_analysis,
                'format_analysis': format_analysis,
                'improvements': improvements,
                'analysis_source': 'local_models'
            }
            
        except Exception as e:
            logger.error(f"Error in resume analysis: {str(e)}")
            return self._get_fallback_analysis(resume_text, job_description)

    def _calculate_basic_match(self, resume_text: str, job_description: str) -> float:
        """Calculate basic match score using keyword matching."""
        resume_words = set(resume_text.lower().split())
        job_words = set(job_description.lower().split())
        
        common_words = resume_words.intersection(job_words)
        match_score = len(common_words) / len(job_words) * 100 if job_words else 0
        
        return match_score

    def _analyze_skills(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Analyze skills in resume and job description."""
        # Common technical skills
        common_skills = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php',
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
            'django', 'flask', 'spring', 'sql', 'nosql', 'mongodb', 'postgresql',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'agile',
            'scrum', 'jira', 'ci/cd', 'devops', 'machine learning', 'ai',
            'data science', 'big data', 'cloud computing'
        ]
        
        # Extract skills
        resume_skills = self._extract_skills(resume_text, common_skills)
        job_skills = self._extract_skills(job_description, common_skills)
        
        # Find matching and missing skills
        matching_skills = list(set(resume_skills).intersection(set(job_skills)))
        missing_skills = list(set(job_skills) - set(resume_skills))
        
        return {
            'matching_skills': matching_skills,
            'missing_skills': missing_skills,
            'skill_gaps': self._identify_skill_gaps(matching_skills, missing_skills)
        }

    def _extract_skills(self, text: str, common_skills: List[str]) -> List[str]:
        """Extract skills from text."""
        text_lower = text.lower()
        found_skills = [skill for skill in common_skills if skill in text_lower]
        
        # Add custom skills found in text
        words = text_lower.split()
        custom_skills = [word for word in words if len(word) > 3 and word not in found_skills]
        
        return found_skills + custom_skills

    def _identify_skill_gaps(self, matching_skills: List[str], missing_skills: List[str]) -> List[str]:
        """Identify skill gaps based on matching and missing skills."""
        return [f"Need to develop skills in: {', '.join(missing_skills[:3])}"]

    def _analyze_format(self, resume_text: str) -> Dict[str, Any]:
        """Analyze resume format and structure."""
        # Basic format analysis
        sections = ['education', 'experience', 'skills', 'projects']
        found_sections = [section for section in sections if section in resume_text.lower()]
        
        score = len(found_sections) / len(sections) * 100
        
        issues = []
        if len(found_sections) < len(sections):
            missing = set(sections) - set(found_sections)
            issues.append(f"Missing sections: {', '.join(missing)}")
        
        suggestions = [
            "Ensure all major sections are present",
            "Use consistent formatting throughout",
            "Include clear section headers"
        ]
        
        return {
            'score': score,
            'issues': issues,
            'suggestions': suggestions
        }

    def _generate_improvements(self, resume_text: str, job_description: str,
                             skills_analysis: Dict[str, Any], format_analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions."""
        improvements = []
        
        # Add improvements based on missing skills
        if skills_analysis['missing_skills']:
            improvements.append(f"Add missing skills: {', '.join(skills_analysis['missing_skills'][:3])}")
        
        # Add improvements based on format issues
        if format_analysis['issues']:
            improvements.extend(format_analysis['issues'])
        
        # Add general improvements
        improvements.extend([
            "Use bullet points for better readability",
            "Include specific achievements with numbers",
            "Tailor your resume to the job description"
        ])
        
        return improvements

    def _get_fallback_analysis(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Get basic analysis when main analysis fails."""
        return {
            'match_score': self._calculate_basic_match(resume_text, job_description),
            'role_analysis': {
                'primary_role': 'Unknown',
                'match_confidence': 0,
                'role_scores': {}
            },
            'skills_analysis': {
                'matching_skills': [],
                'missing_skills': [],
                'skill_gaps': ['Basic analysis only']
            },
            'format_analysis': {
                'score': 50,
                'issues': ['Basic analysis only'],
                'suggestions': ['Try again later for detailed analysis']
            },
            'improvements': ['Please try again later for detailed analysis'],
            'analysis_source': 'fallback'
        }

# Initialize global analyzer instance
_analyzer = None

def get_analyzer() -> ResumeAnalyzer:
    """Get the global resume analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = ResumeAnalyzer()
    return _analyzer 