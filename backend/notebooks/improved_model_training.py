import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

def remove_outliers(df, columns):
    """Remove outliers using IQR method"""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def train_model(pairs_df):
    """Train the resume-job matching model with improved regularization"""
    print("Preparing features for model training...")
    
    model_data = pairs_df.copy()
    
    # Add regularization features
    model_data['skill_count_diff'] = abs(model_data['resume_skills_count'] - model_data['job_skills_count'])
    model_data['skill_count_ratio'] = model_data['common_skills_count'] / model_data['job_skills_count'].clip(lower=1)
    
    # Select features with domain knowledge
    features = [
        'jaccard_similarity',
        'common_skills_count', 
        'skill_vector_similarity',
        'skill_count_diff',
        'skill_count_ratio',
        'skills_density'
    ]
    
    # Remove outliers
    model_data = remove_outliers(model_data, features)
    model_data = model_data.dropna(subset=features)
    
    X = model_data[features].values
    y = model_data['match_score'].values
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=pd.qcut(y, q=5, labels=False)  # Stratify by match score quintiles
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training models with cross-validation...")
    
    # Define hyperparameter grids with regularization
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt', 'log2']
    }
    
    # Train with cross-validation
    rf = RandomForestRegressor(random_state=42)
    rf_grid = GridSearchCV(
        rf, 
        rf_params, 
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    rf_grid.fit(X_train_scaled, y_train)
    
    gb = GradientBoostingRegressor(random_state=42)
    gb_grid = GridSearchCV(
        gb, 
        gb_params, 
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    gb_grid.fit(X_train_scaled, y_train)
    
    # Get best models
    rf_best = rf_grid.best_estimator_
    gb_best = gb_grid.best_estimator_
    
    # Evaluate on test set
    rf_pred = rf_best.predict(X_test_scaled)
    gb_pred = gb_best.predict(X_test_scaled)
    
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    
    print(f"Random Forest RMSE: {rf_rmse:.4f}")
    print(f"Gradient Boosting RMSE: {gb_rmse:.4f}")
    
    # Select best model based on test performance
    if rf_rmse <= gb_rmse:
        best_model = rf_best
        best_pred = rf_pred
        print("Random Forest selected as best model")
    else:
        best_model = gb_best
        best_pred = gb_pred
        print("Gradient Boosting selected as best model")
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    print(f"\nCross-validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Evaluation metrics
    print("\nModel Evaluation:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, best_pred)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, best_pred):.4f}")
    print(f"R² Score: {r2_score(y_test, best_pred):.4f}")
    
    # Visualize predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, best_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Match Score')
    plt.ylabel('Predicted Match Score')
    plt.title('Predicted vs Actual Match Scores')
    plt.tight_layout()
    plt.savefig('match_score_prediction.png')
    plt.show()
    
    # Save the model and scaler
    joblib.dump(best_model, 'resume_job_matching_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    joblib.dump(features, 'model_features.pkl')
    
    print("\nModel saved as 'resume_job_matching_model.pkl'")
    print("Feature scaler saved as 'feature_scaler.pkl'")
    print("Training complete!")
    
    return best_model, scaler, features, feature_importance 