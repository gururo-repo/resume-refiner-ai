import os
import pickle
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
import joblib
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from dotenv import load_dotenv
from .groq_analyzer import get_groq_analysis
from sklearn.metrics.pairwise import cosine_similarity
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
torch.hub.set_dir(CACHE_DIR)

# Configure requests session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Global variables for singleton pattern
_loaded_models = None
_match_model = None
_feature_scaler = None
_model_features = None
_sentence_transformer = None
_genai_model = None
_model_loader = None

class ModelLoader:
    """Handles loading and management of models with memory optimization."""
    
    def __init__(self):
        """Initialize the model loader."""
        self.sentence_transformer = None
        self.initialized = False
        self.use_groq = True  # Start with Groq enabled
        self._initialize_models()

    def _initialize_models(self):
        """Initialize only essential models."""
        try:
            # Initialize sentence transformer with memory optimization
            self.sentence_transformer = self._load_sentence_transformer()
            self.initialized = True
            logger.info("Successfully initialized models")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            self.initialized = False

    def _load_sentence_transformer(self) -> Optional[SentenceTransformer]:
        """Load the sentence transformer model with memory optimization."""
        try:
            # Set environment variables for memory optimization
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = CACHE_DIR
            os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
            os.environ['HF_HOME'] = CACHE_DIR
            
            # Force CPU usage to reduce memory consumption
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            # Load model with memory optimization
            model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device='cpu',
                cache_folder=CACHE_DIR
            )
            
            # Test the model
            test_text = "Testing model initialization"
            model.encode(test_text)
            
            logger.info("Successfully loaded sentence transformer")
            return model
        except Exception as e:
            logger.error(f"Error in sentence transformer initialization: {str(e)}")
            return None

    def get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embeddings for a list of texts with memory optimization."""
        if not self.initialized or self.sentence_transformer is None:
            logger.warning("Sentence transformer not available, using basic embeddings")
            return self._get_basic_embeddings(texts)

        try:
            # Process in smaller batches to reduce memory usage
            batch_size = 8
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.sentence_transformer.encode(batch)
                all_embeddings.append(batch_embeddings)
                
            return np.vstack(all_embeddings)
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            return self._get_basic_embeddings(texts)

    def _get_basic_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate basic embeddings when the model is not available."""
        all_words = set()
        for text in texts:
            all_words.update(text.lower().split())
        
        word_to_idx = {word: idx for idx, word in enumerate(all_words)}
        embeddings = np.zeros((len(texts), len(word_to_idx)))
        
        for i, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                if word in word_to_idx:
                    embeddings[i, word_to_idx[word]] += 1
        
        # Normalize embeddings
        row_sums = embeddings.sum(axis=1)
        embeddings = embeddings / row_sums[:, np.newaxis]
        
        return embeddings

    def try_groq_analysis(self, resume_text: str, job_description: str) -> Optional[Dict[str, Any]]:
        """Try to get analysis from Groq, with fallback to local models if it fails."""
        if not self.use_groq:
            logger.info("Groq analysis disabled, using local models")
            return None

        try:
            analysis = get_groq_analysis(resume_text, job_description)
            if analysis:
                logger.info("Successfully got analysis from Groq")
                return analysis
            else:
                logger.warning("Groq analysis returned empty result")
                self.use_groq = False
                return None
        except Exception as e:
            logger.error(f"Error in Groq analysis: {str(e)}")
            self.use_groq = False
            return None

# Initialize global model loader instance
_model_loader = None

def get_model_loader() -> ModelLoader:
    """Get or create the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader

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

def get_sentence_transformer() -> SentenceTransformer:
    """Get or initialize the global sentence transformer instance."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            # Suppress unnecessary warnings
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            # Set environment variables to optimize loading
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = CACHE_DIR
            os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
            os.environ['HF_HOME'] = CACHE_DIR
            
            # Create cache directory if it doesn't exist
            Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
            
            # Try loading with increased timeout and retries
            max_retries = 5
            retry_delay = 10  # seconds
            timeout = 30  # seconds
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to load sentence transformer model (attempt {attempt + 1}/{max_retries})...")
                    
                    # Configure model loading with increased timeout
                    _sentence_transformer = SentenceTransformer(
                        'all-MiniLM-L6-v2',
                        device='cpu',
                        cache_folder=CACHE_DIR,
                        show_progress_bar=False,
                        timeout=timeout
                    )
                    
                    # Test the model with a simple encoding
                    test_text = "Testing model initialization"
                    _sentence_transformer.encode(test_text, show_progress_bar=False)
                    
                    logger.info("Successfully initialized sentence transformer model")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        # Increase timeout for next attempt
                        timeout *= 2
                    else:
                        logger.error(f"All attempts to load model failed: {str(e)}")
                        raise
            
        except Exception as e:
            logger.error(f"Error initializing sentence transformer: {str(e)}")
            raise
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