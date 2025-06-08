import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import joblib
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """Load and preprocess the datasets."""
    try:
        # Load datasets
        job_df = pd.read_csv('data/IT_Job_Roles_Skills.csv', encoding='latin1')
        resume_df = pd.read_csv('data/UpdatedResumeDataSet.csv', encoding='latin1')
        
        # Clean skills
        def clean_skills(skill_str):
            if pd.isna(skill_str):
                return []
            return [skill.strip().lower() for skill in skill_str.split(',')]
        
        job_df['Skills_clean'] = job_df['Skills'].apply(clean_skills)
        job_df['Job Title'] = job_df['Job Title'].str.strip().str.lower()
        job_df['Job Description'] = job_df['Job Description'].fillna('').str.lower()
        
        resume_df = resume_df.dropna(subset=['Resume'])
        resume_df['Resume'] = resume_df['Resume'].str.lower()
        
        if 'Skills' in resume_df.columns:
            resume_df['Skills_clean'] = resume_df['Skills'].apply(clean_skills)
        
        return job_df, resume_df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_training_pairs(job_df, resume_df):
    """Create training pairs for model training."""
    try:
        # Initialize Sentence Transformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create skill vectors
        all_job_skills = set(skill for skills_list in job_df['Skills_clean'] for skill in skills_list)
        skill_vocabulary = sorted(list(all_job_skills))
        
        def create_skill_vector(skills_list):
            indices = [skill_vocabulary.index(skill) for skill in skills_list if skill in skill_vocabulary]
            if not indices:
                return np.zeros(len(skill_vocabulary))
            vector = np.zeros(len(skill_vocabulary))
            vector[indices] = 1
            return vector
        
        # Create pairs
        pairs = []
        n_resumes = len(resume_df)
        n_jobs = len(job_df)
        max_pairs = 10000
        sample_ratio = min(1.0, max_pairs / (n_resumes * n_jobs))
        
        if sample_ratio < 1.0:
            for _ in range(max_pairs):
                resume_idx = np.random.randint(0, n_resumes)
                job_idx = np.random.randint(0, n_jobs)
                
                resume_skills = set(resume_df.iloc[resume_idx]['Skills_clean'])
                job_skills = set(job_df.iloc[job_idx]['Skills_clean'])
                
                jaccard_similarity = len(resume_skills.intersection(job_skills)) / len(resume_skills.union(job_skills)) if resume_skills and job_skills else 0.0
                common_skills_count = len(resume_skills.intersection(job_skills))
                
                resume_vec = create_skill_vector(resume_skills)
                job_vec = create_skill_vector(job_skills)
                skill_vector_similarity = np.dot(resume_vec, job_vec) / (np.linalg.norm(resume_vec) * np.linalg.norm(job_vec)) if np.any(resume_vec) and np.any(job_vec) else 0.0
                
                match_score = (
                    0.3 * jaccard_similarity +
                    0.2 * (common_skills_count / max(1, len(job_skills))) +
                    0.5 * skill_vector_similarity
                )
                
                pairs.append({
                    'resume_idx': resume_idx,
                    'job_idx': job_idx,
                    'jaccard_similarity': jaccard_similarity,
                    'common_skills_count': common_skills_count,
                    'skill_vector_similarity': skill_vector_similarity,
                    'match_score': match_score,
                    'skills_density': len(resume_skills) / max(1, len(resume_df.iloc[resume_idx]['Resume'].split())),
                    'resume_skills_count': len(resume_skills),
                    'job_skills_count': len(job_skills)
                })
        
        return pd.DataFrame(pairs)
        
    except Exception as e:
        logger.error(f"Error creating training pairs: {str(e)}")
        raise

def train_models(pairs_df):
    """Train and save the models."""
    try:
        # Prepare features
        features = [
            'jaccard_similarity',
            'common_skills_count',
            'skill_vector_similarity',
            'skills_density',
            'resume_skills_count',
            'job_skills_count'
        ]
        
        X = pairs_df[features].values
        y = pairs_df['match_score'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestRegressor(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        rf_grid.fit(X_train_scaled, y_train)
        
        # Train Gradient Boosting
        gb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
        
        gb = GradientBoostingRegressor(random_state=42)
        gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        gb_grid.fit(X_train_scaled, y_train)
        
        # Select best model
        rf_best = rf_grid.best_estimator_
        gb_best = gb_grid.best_estimator_
        
        rf_pred = rf_best.predict(X_test_scaled)
        gb_pred = gb_best.predict(X_test_scaled)
        
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
        
        best_model = rf_best if rf_rmse <= gb_rmse else gb_best
        
        # Save models
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/job_matching_model.pkl')
        joblib.dump(scaler, 'models/job_scaler.pkl')
        joblib.dump(features, 'models/job_features.pkl')
        
        # Print evaluation metrics
        best_pred = rf_pred if rf_rmse <= gb_rmse else gb_pred
        print(f"\nModel Evaluation:")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, best_pred)):.4f}")
        print(f"MAE: {mean_absolute_error(y_test, best_pred):.4f}")
        print(f"R² Score: {r2_score(y_test, best_pred):.4f}")
        
        return best_model, scaler, features
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise

def main():
    """Main function to train and save models."""
    try:
        logger.info("Loading and preprocessing data...")
        job_df, resume_df = load_and_preprocess_data()
        
        logger.info("Creating training pairs...")
        pairs_df = create_training_pairs(job_df, resume_df)
        
        logger.info("Training models...")
        best_model, scaler, features = train_models(pairs_df)
        
        logger.info("Models trained and saved successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main() 