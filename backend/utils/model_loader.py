import os
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Define model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')

# Load models
try:
    match_model = joblib.load(os.path.join(MODEL_DIR, 'resume_job_matching_model.pkl'))
    feature_scaler = joblib.load(os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
    model_features = joblib.load(os.path.join(MODEL_DIR, 'model_features.pkl'))
    
    # Initialize sentence transformer for embeddings
    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    
    print("Successfully loaded all models!")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise 