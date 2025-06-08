import logging
from typing import Dict, Any
from sentence_transformers import SentenceTransformer, util
import torch
import joblib
import os

logger = logging.getLogger(__name__)

class ATSScorer:
    def __init__(self):
        self.model = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the Sentence-BERT model."""
        try:
            model_path = os.path.join('models', 'ats_scorer')
            if os.path.exists(model_path):
                logger.info("Loading fine-tuned ATS scorer model...")
                self.model = joblib.load(model_path)
            else:
                logger.info("Loading base Sentence-BERT model...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error initializing ATS scorer model: {str(e)}")
            raise
    
    def calculate_score(self, resume_data: Dict[str, Any], job_description: str) -> float:
        """
        Calculate ATS compatibility score between resume and job description.
        
        Args:
            resume_data: Parsed resume data
            job_description: Job description text
            
        Returns:
            ATS compatibility score (0-100)
        """
        try:
            # Combine relevant sections for comparison
            resume_text = self._prepare_resume_text(resume_data)
            
            # Encode texts
            resume_embedding = self.model.encode(resume_text, convert_to_tensor=True)
            job_embedding = self.model.encode(job_description, convert_to_tensor=True)
            
            # Calculate similarity
            similarity = util.pytorch_cos_sim(resume_embedding, job_embedding)
            
            # Convert to percentage
            score = float(similarity[0][0]) * 100
            
            return round(score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating ATS score: {str(e)}")
            raise
    
    def _prepare_resume_text(self, resume_data: Dict[str, Any]) -> str:
        """Prepare resume text for comparison by combining relevant sections."""
        sections = []
        
        # Add skills section
        if 'skills' in resume_data:
            sections.append(' '.join(resume_data['skills']))
        
        # Add experience section
        if 'experience' in resume_data['sections']:
            sections.append(resume_data['sections']['experience'])
        
        # Add education section
        if 'education' in resume_data['sections']:
            sections.append(resume_data['sections']['education'])
        
        return ' '.join(sections)

# Initialize global scorer instance
ats_scorer = ATSScorer()

def calculate_ats_score(resume_data: Dict[str, Any], job_description: str) -> float:
    """
    Calculate ATS compatibility score.
    
    Args:
        resume_data: Parsed resume data
        job_description: Job description text
        
    Returns:
        ATS compatibility score (0-100)
    """
    return ats_scorer.calculate_score(resume_data, job_description) 