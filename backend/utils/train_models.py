import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv_with_encoding(file_path):
    """Try different encodings to load CSV file"""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            logger.info(f"Trying to load {file_path} with {encoding} encoding")
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error loading {file_path} with {encoding}: {str(e)}")
            continue
    
    raise ValueError(f"Could not load {file_path} with any of the attempted encodings")

def train_and_save_models():
    """
    Train and save models compatible with Python 3.13.4
    """
    try:
        # Load the training data
        notebooks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'notebooks')
        
        # Load CSV files with proper encoding
        resume_df = load_csv_with_encoding(os.path.join(notebooks_dir, 'UpdatedResumeDataSet.csv'))
        job_df = load_csv_with_encoding(os.path.join(notebooks_dir, 'IT_Job_Roles_Skills.csv'))
        
        logger.info("Successfully loaded CSV files")
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Extract features from the data
        logger.info("Extracting features from data...")
        
        # For now, we'll use a simple feature set
        X = np.random.rand(100, 5)  # Example features
        y = np.random.rand(100)     # Example target
        
        # Train the model
        logger.info("Training RandomForest model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Create and fit a scaler
        logger.info("Creating and fitting scaler...")
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Save the models with protocol 5 (highest protocol)
        logger.info("Saving models...")
        with open(os.path.join(models_dir, 'resume_job_matching_model.pkl'), 'wb') as f:
            pickle.dump(model, f, protocol=5)
        
        with open(os.path.join(models_dir, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f, protocol=5)
        
        with open(os.path.join(models_dir, 'model_features.pkl'), 'wb') as f:
            pickle.dump(['feature1', 'feature2', 'feature3', 'feature4', 'feature5'], f, protocol=5)
        
        logger.info("Successfully trained and saved new models")
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_models() 