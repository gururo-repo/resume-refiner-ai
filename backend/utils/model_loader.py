import os
import pickle
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
import joblib
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
torch.hub.set_dir(CACHE_DIR)

# Global variables for singleton pattern
_loaded_models = None
_match_model = None
_feature_scaler = None
_model_features = None
_sentence_transformer = None
_genai_model = None

class ModelLoader:
    def __init__(self):
        self.models_dir = os.path.join('models')
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all ML models and GenAI."""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Initialize GenAI
            self._initialize_genai()
            
            # Initialize Sentence Transformer for fallback
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load or initialize job matching model (fallback)
            self.job_matcher = self._load_or_create_model('job_matching_model.pkl')
            self.job_scaler = self._load_or_create_model('job_scaler.pkl')
            self.job_features = self._load_or_create_model('job_features.pkl')
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def _initialize_genai(self):
        """Initialize Google Generative AI."""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.warning("Google API key not found. GenAI features will be disabled.")
                return
            
            genai.configure(api_key=api_key)
            
            # List available models and their capabilities
            models = genai.list_models()
            available_models = [model.name for model in models]
            logger.info(f"Available GenAI models: {available_models}")
            
            # Try to use gemini-1.5-pro first
            model = "models/gemini-1.5-pro"
            url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
            
            if model in available_models:
                self.genai_model = genai.GenerativeModel(
                    model_name=model,
                    generation_config={
                        'temperature': 0.7,
                        'top_p': 0.8,
                        'top_k': 40,
                        'max_output_tokens': 2048,
                    }
                )
                logger.info(f"Successfully initialized GenAI model: {model}")
                return
            
            # Fallback to gemini-1.0-pro
            model = "models/gemini-1.0-pro"
            url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
            
            if model in available_models:
                self.genai_model = genai.GenerativeModel(
                    model_name=model,
                    generation_config={
                        'temperature': 0.7,
                        'top_p': 0.8,
                        'top_k': 40,
                        'max_output_tokens': 2048,
                    }
                )
                logger.info(f"Successfully initialized GenAI model: {model}")
                return
            
            logger.warning("No suitable GenAI model found. GenAI features will be disabled.")
            self.genai_model = None
            
        except Exception as e:
            logger.error(f"Error initializing GenAI: {str(e)}")
            self.genai_model = None
    
    def _load_or_create_model(self, model_name: str):
        """Load existing model or return None if not found."""
        try:
            model_path = os.path.join(self.models_dir, model_name)
            if os.path.exists(model_path):
                return joblib.load(model_path)
            return None
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return None
    
    def predict_job_match(self, resume_data: Dict[str, Any], job_description: str) -> float:
        """Predict job match score using GenAI first, then fallback to ML model."""
        try:
            # Try GenAI first
            if self.genai_model:
                try:
                    match_score = self._get_genai_match_score(resume_data, job_description)
                    if match_score is not None:
                        return match_score
                except Exception as e:
                    logger.warning(f"GenAI prediction failed, falling back to ML model: {str(e)}")
            
            # Fallback to ML model
            if self.job_matcher and self.job_scaler and self.job_features:
                try:
                    features = self._prepare_job_match_features(resume_data, job_description)
                    features_scaled = self.job_scaler.transform(features.reshape(1, -1))
                    match_score = self.job_matcher.predict(features_scaled)[0]
                    return float(match_score)
                except Exception as e:
                    logger.warning(f"ML model prediction failed, falling back to basic match: {str(e)}")
            
            # Final fallback to basic matching
            return self._calculate_basic_match(resume_data, job_description)
            
        except Exception as e:
            logger.error(f"Error predicting job match: {str(e)}")
            return 0.0
    
    def _get_genai_match_score(self, resume_data: Dict[str, Any], job_description: str) -> Optional[float]:
        """Get job match score using GenAI."""
        try:
            if not self.genai_model:
                logger.warning("GenAI model not initialized")
                return None

            prompt = self._prepare_match_prompt(resume_data, job_description)
            response = self.genai_model.generate_content(prompt)
            
            try:
                # Parse response to get match score
                response_text = response.text.strip()
                # Extract score from response
                if 'match score:' in response_text.lower():
                    score_text = response_text.lower().split('match score:')[1].strip()
                    # Convert percentage to decimal if needed
                    if '%' in score_text:
                        score = float(score_text.replace('%', '')) / 100
                    else:
                        score = float(score_text)
                    return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
                else:
                    # Try to find any number in the response
                    import re
                    numbers = re.findall(r'\d+\.?\d*', response_text)
                    if numbers:
                        score = float(numbers[0])
                        if score > 1:  # If it's a percentage
                            score = score / 100
                        return min(max(score, 0.0), 1.0)
                    return None
            except (ValueError, AttributeError) as e:
                logger.warning(f"Could not parse GenAI response as score: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting GenAI match score: {str(e)}")
            return None
    
    def _prepare_match_prompt(self, resume_data: Dict[str, Any], job_description: str) -> str:
        """Prepare prompt for GenAI match score prediction."""
        resume_text = self._prepare_resume_text(resume_data)
        
        prompt = f"""Analyze the match between this resume and job description.
        Resume:
        {resume_text}
        
        Job Description:
        {job_description}
        
        Based on the above, provide a detailed analysis including:
        1. Match Score: [provide a score between 0 and 1]
        2. Strengths: [list key strengths that match the job requirements]
        3. Weaknesses: [list areas that need improvement]
        4. Recommendations: [suggest specific improvements]
        
        Format your response with clear sections and bullet points."""
        
        return prompt
    
    def _prepare_job_match_features(self, resume_data: Dict[str, Any], job_description: str) -> np.ndarray:
        """Prepare features for job matching."""
        # Extract skills
        resume_skills = set(resume_data.get('skills', []))
        job_skills = self._extract_skills_from_text(job_description)
        
        # Calculate metrics
        jaccard_similarity = self._calculate_jaccard_similarity(resume_skills, job_skills)
        common_skills_count = len(resume_skills.intersection(job_skills))
        skill_vector_similarity = self._calculate_skill_vector_similarity(resume_data, job_description)
        
        # Calculate additional features
        skill_count_diff = abs(len(resume_skills) - len(job_skills))
        skill_count_ratio = common_skills_count / max(1, len(job_skills))
        skills_density = len(resume_skills) / max(1, len(resume_data.get('raw_text', '').split()))
        
        return np.array([
            jaccard_similarity,
            common_skills_count,
            skill_vector_similarity,
            skill_count_diff,
            skill_count_ratio,
            skills_density
        ])
    
    def _calculate_basic_match(self, resume_data: Dict[str, Any], job_description: str) -> float:
        """Calculate basic match score when ML model is not available."""
        try:
            # Extract skills
            resume_skills = set(resume_data.get('skills', []))
            job_skills = self._extract_skills_from_text(job_description)
            
            # Calculate metrics
            jaccard_similarity = self._calculate_jaccard_similarity(resume_skills, job_skills)
            common_skills_count = len(resume_skills.intersection(job_skills))
            skill_vector_similarity = self._calculate_skill_vector_similarity(resume_data, job_description)
            
            # Calculate weighted score with adjusted weights
            match_score = (
                0.4 * jaccard_similarity +  # Increased weight for skill match
                0.3 * (common_skills_count / max(1, len(job_skills))) +  # Increased weight for common skills
                0.3 * skill_vector_similarity  # Reduced weight for semantic similarity
            )
            
            # Scale the score to be more lenient
            scaled_score = match_score * 1.5  # Increase scores by 50%
            return min(float(scaled_score), 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating basic match: {str(e)}")
            return 0.0
    
    def _calculate_jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def _calculate_skill_vector_similarity(self, resume_data: Dict[str, Any], job_description: str) -> float:
        """Calculate semantic similarity between resume and job description."""
        try:
            # Combine relevant sections
            resume_text = self._prepare_resume_text(resume_data)
            
            # Encode texts
            resume_embedding = self.sentence_model.encode(resume_text, convert_to_tensor=True)
            job_embedding = self.sentence_model.encode(job_description, convert_to_tensor=True)
            
            # Calculate similarity
            similarity = np.dot(resume_embedding, job_embedding) / (
                np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating skill vector similarity: {str(e)}")
            return 0.0
    
    def _prepare_resume_text(self, resume_data: Dict[str, Any]) -> str:
        """Prepare resume text for analysis."""
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

# Initialize global model loader instance
model_loader = ModelLoader()

def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    return model_loader

def load_models():
    """
    Load all models and components from the models directory.
    This function is called only once when first needed.
    """
    global _loaded_models, _match_model, _feature_scaler, _model_features, _sentence_transformer, _genai_model
    
    if _loaded_models is not None:
        return _loaded_models
    
    try:
        # Get the models directory path
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        # Check if models directory exists
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
            logger.info(f"Created models directory at {models_dir}")
        
        # Initialize sentence transformer with cache handling
        try:
            _sentence_transformer = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder=CACHE_DIR,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            logger.info("Successfully initialized sentence transformer")
        except Exception as e:
            logger.error(f"Failed to initialize sentence transformer: {str(e)}")
            _sentence_transformer = None
        
        # Initialize GenAI
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                _genai_model = genai.GenerativeModel('gemini-pro')
                logger.info("Successfully initialized GenAI model")
            else:
                logger.warning("Google API key not found. GenAI features will be disabled.")
                _genai_model = None
        except Exception as e:
            logger.error(f"Failed to initialize GenAI: {str(e)}")
            _genai_model = None
        
        # Create the models dictionary
        _loaded_models = {
            'sentence_transformer': _sentence_transformer,
            'genai_model': _genai_model
        }
        
        logger.info("Successfully loaded all models")
        return _loaded_models
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return None

def get_models():
    """
    Get the loaded models dictionary.
    This is the main entry point for accessing models.
    """
    return load_models()

def get_match_model():
    """
    Get the resume job matching model.
    """
    if _match_model is None:
        load_models()
    return _match_model

def get_feature_scaler():
    """
    Get the feature scaler.
    """
    if _feature_scaler is None:
        load_models()
    return _feature_scaler

def get_model_features():
    """
    Get the model features.
    """
    if _model_features is None:
        load_models()
    return _model_features

def get_sentence_transformer():
    """
    Get the sentence transformer model.
    """
    if _sentence_transformer is None:
        load_models()
    return _sentence_transformer

def get_genai_model():
    """
    Get the GenAI model.
    """
    if _genai_model is None:
        load_models()
    return _genai_model

__all__ = [
    'get_models',
    'get_match_model',
    'get_feature_scaler',
    'get_model_features',
    'get_sentence_transformer',
    'get_genai_model'
] 