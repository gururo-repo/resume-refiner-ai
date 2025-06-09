import logging
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer, util
import torch
import joblib
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .model_loader import get_sentence_transformer

logger = logging.getLogger(__name__)

class ATSScorer:
    def __init__(self):
        """Initialize the ATS scorer."""
        self.model = None
        self.initialized = False
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            logger.info("Getting sentence transformer model...")
            self.model = get_sentence_transformer()
            self.initialized = True
            logger.info("Successfully initialized ATS scorer")
        except Exception as e:
            logger.error(f"Error initializing ATS scorer: {str(e)}")
            self.initialized = False
            logger.info("ATS scorer will use basic scoring methods")

    def calculate_ats_score(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Calculate ATS score for a resume against a job description.
        
        Args:
            resume_text: The resume text to analyze
            job_description: The job description text
            
        Returns:
            Dictionary containing ATS score and analysis
        """
        if not self.initialized:
            logger.warning("ATS scorer not initialized, using basic scoring")
            return self._basic_ats_score(resume_text, job_description)

        try:
            # Get embeddings in a single batch
            texts = [resume_text, job_description]
            embeddings = self.model.encode(texts, show_progress_bar=False)
            resume_embedding, job_embedding = embeddings

            # Calculate similarity
            similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
            ats_score = float(similarity * 100)

            # Analyze format and content
            format_analysis = self._analyze_format(resume_text)
            content_analysis = self._analyze_content(resume_text, job_description)

            return {
                'ats_score': ats_score,
                'format_analysis': format_analysis,
                'content_analysis': content_analysis
            }
        except Exception as e:
            logger.error(f"Error calculating ATS score: {str(e)}")
            return self._basic_ats_score(resume_text, job_description)

    def _basic_ats_score(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Calculate basic ATS score when model is not available."""
        # Simple keyword matching
        resume_words = set(resume_text.lower().split())
        job_words = set(job_description.lower().split())
        
        # Calculate basic match score
        common_words = resume_words.intersection(job_words)
        match_score = len(common_words) / len(job_words) * 100 if job_words else 0

        return {
            'ats_score': match_score * 0.8,  # ATS score is typically lower than match score
            'format_analysis': {
                'score': 50,
                'issues': ['Basic analysis only'],
                'suggestions': ['Try again later for detailed format analysis']
            },
            'content_analysis': {
                'keyword_match': len(common_words),
                'missing_keywords': list(job_words - resume_words),
                'suggestions': ['Add missing keywords to improve ATS score']
            }
        }

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

# Initialize global ATS scorer instance
_ats_scorer = None

def get_ats_scorer() -> ATSScorer:
    """Get the global ATS scorer instance."""
    global _ats_scorer
    if _ats_scorer is None:
        _ats_scorer = ATSScorer()
    return _ats_scorer

def calculate_ats_score(resume_data: Dict[str, Any], job_description: str) -> float:
    """
    Calculate ATS compatibility score.
    
    Args:
        resume_data: Parsed resume data
        job_description: Job description text
        
    Returns:
        ATS compatibility score (0-100)
    """
    return get_ats_scorer().calculate_score(resume_data, job_description) 