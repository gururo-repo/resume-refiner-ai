import os
import pickle
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for singleton pattern
_loaded_models = None
_match_model = None
_feature_scaler = None
_model_features = None
_sentence_transformer = None

def load_models():
    """
    Load all models and components from the models directory.
    This function is called only once when first needed.
    """
    global _loaded_models, _match_model, _feature_scaler, _model_features, _sentence_transformer
    
    if _loaded_models is not None:
        return _loaded_models
    
    try:
        # Get the models directory path
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        # Check if models directory exists
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found at {models_dir}")
        
        # Check if all required model files exist
        required_files = [
            'resume_job_matching_model.pkl',
            'feature_scaler.pkl',
            'model_features.pkl'
        ]
        
        for file in required_files:
            file_path = os.path.join(models_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required model file not found: {file}")
        
        # Load the models
        with open(os.path.join(models_dir, 'resume_job_matching_model.pkl'), 'rb') as f:
            _match_model = pickle.load(f)
            logger.info("Successfully loaded resume job matching model")
        
        with open(os.path.join(models_dir, 'feature_scaler.pkl'), 'rb') as f:
            _feature_scaler = pickle.load(f)
            logger.info("Successfully loaded feature scaler")
        
        with open(os.path.join(models_dir, 'model_features.pkl'), 'rb') as f:
            _model_features = pickle.load(f)
            logger.info("Successfully loaded model features")
        
        # Initialize sentence transformer
        _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Successfully initialized sentence transformer")
        
        # Create the models dictionary
        _loaded_models = {
            'match_model': _match_model,
            'feature_scaler': _feature_scaler,
            'model_features': _model_features,
            'sentence_transformer': _sentence_transformer
        }
        
        logger.info("Successfully loaded all models")
        return _loaded_models
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

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

__all__ = [
    'get_models',
    'get_match_model',
    'get_feature_scaler',
    'get_model_features',
    'get_sentence_transformer'
] 