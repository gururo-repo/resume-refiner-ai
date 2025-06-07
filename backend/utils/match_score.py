import json
import numpy as np
import logging
from .genai_suggester import ResumeImprover
from .model_loader import get_models, get_match_model, get_feature_scaler, get_model_features, get_sentence_transformer
from .skill_matcher import extract_skills, SKILLS_DB
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchScoreCalculator:
    def __init__(self):
        try:
            self.models = get_models()
            self.match_model = get_match_model()
            self.feature_scaler = get_feature_scaler()
            self.model_features = get_model_features()
            self.sentence_transformer = get_sentence_transformer()
            logger.info("Successfully initialized MatchScoreCalculator")
        except Exception as e:
            logger.error(f"Failed to initialize MatchScoreCalculator: {str(e)}")
            raise

    def calculate_match_score(self, resume_skills, job_skills, resume_text, job_description):
        """
        Calculate match score between resume and job description.
        First attempts LLM-based analysis, falls back to model-based analysis if LLM fails.
        """
        try:
            # First attempt: Use LLM-based analysis
            suggester = ResumeImprover()
            analysis = suggester.analyze_ats_compatibility(resume_text, job_description)
            
            # Extract scores from LLM analysis
            match_score = analysis.get('match_score', 0)
            ats_score = analysis.get('ats_score', 0)
            strengths = analysis.get('strengths', [])
            weaknesses = analysis.get('weaknesses', [])
            recommendations = analysis.get('recommendations', [])
            
            logger.info("Successfully completed LLM-based analysis")
            return {
                'match_score': match_score,
                'ats_score': ats_score,
                'strengths': strengths,
                'weaknesses': weaknesses,
                'recommendations': recommendations,
                'analysis_type': 'llm'
            }
            
        except Exception as e:
            logger.warning(f"LLM analysis failed, falling back to model-based analysis: {str(e)}")
            return self._model_based_analysis(resume_skills, job_skills, resume_text)

    def _model_based_analysis(self, resume_skills, job_skills, resume_text):
        """
        Fallback method using trained model for match analysis.
        Only used when LLM analysis fails.
        """
        try:
            # Calculate features using the same methodology as training
            jaccard_similarity = self._calculate_jaccard_similarity(resume_skills, job_skills)
            common_skills_count = len(resume_skills.intersection(job_skills))
            skill_vector_similarity = self._calculate_skill_vector_similarity(resume_skills, job_skills)
            skill_count_diff = abs(len(resume_skills) - len(job_skills))
            skill_count_ratio = common_skills_count / max(1, len(job_skills))
            skills_density = len(resume_skills) / max(1, len(resume_text.split()))

            # Create feature array matching the model's expected input
            features = np.array([[
                jaccard_similarity,
                common_skills_count,
                skill_vector_similarity,
                skill_count_diff,
                skill_count_ratio,
                skills_density
            ]])

            # Scale features using the saved scaler
            features_scaled = self.feature_scaler.transform(features)

            # Predict match score using the trained model
            match_score = self.match_model.predict(features_scaled)[0]
            
            # Calculate ATS score based on model features
            ats_score = min(100, max(0, (
                jaccard_similarity * 0.4 +
                skill_vector_similarity * 0.3 +
                skill_count_ratio * 0.3
            ) * 100))
            
            # Generate strengths and weaknesses based on model analysis
            strengths = [f"Strong skill: {skill}" for skill in list(resume_skills.intersection(job_skills))[:5]]
            weaknesses = [f"Missing skill: {skill}" for skill in list(job_skills - resume_skills)[:5]]
            
            # Generate recommendations based on model analysis
            recommendations = []
            if weaknesses:
                recommendations.append(f"Add missing skills: {', '.join(list(job_skills - resume_skills)[:3])}")
            if skills_density < 0.1:
                recommendations.append("Increase skills density in resume")
            if len(resume_text.split()) < 200:
                recommendations.append("Add more detailed content to resume")
            
            logger.info("Successfully completed model-based analysis")
            return {
                'match_score': max(0, min(100, match_score * 100)),
                'ats_score': ats_score,
                'strengths': strengths,
                'weaknesses': weaknesses,
                'recommendations': recommendations,
                'analysis_type': 'model'
            }
            
        except Exception as e:
            logger.error(f"Model-based analysis failed: {str(e)}")
            return {
                'match_score': 0,
                'ats_score': 0,
                'strengths': ['Basic resume structure present'],
                'weaknesses': ['Unable to perform detailed analysis'],
                'recommendations': ['Please try again or contact support'],
                'analysis_type': 'fallback'
            }

    def _calculate_jaccard_similarity(self, set1, set2):
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _calculate_skill_vector_similarity(self, resume_skills, job_skills):
        if not resume_skills or not job_skills:
            return 0.0
        
        # Create binary vectors for skills
        all_skills = list(self.model_features)
        resume_vec = np.array([1 if skill in resume_skills else 0 for skill in all_skills])
        job_vec = np.array([1 if skill in job_skills else 0 for skill in all_skills])
        
        # Calculate cosine similarity
        return cosine_similarity([resume_vec], [job_vec])[0][0]

# Create a singleton instance
match_score_calculator = MatchScoreCalculator()

def get_match_score(resume_skills, job_skills, resume_text, job_description):
    """
    Get match score between resume and job description.
    This is the main entry point for match score calculation.
    """
    return match_score_calculator.calculate_match_score(resume_skills, job_skills, resume_text, job_description)

def _model_based_analysis(resume_text, job_description):
    """
    Fallback method using trained model for match analysis.
    """
    try:
        # Extract skills from both texts
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)
        
        # Calculate skill match metrics
        resume_skill_set = set()
        job_skill_set = set()
        
        for category, skills in resume_skills.items():
            resume_skill_set.update(skills)
        
        for category, skills in job_skills.items():
            job_skill_set.update(skills)
        
        # Calculate Jaccard similarity
        if len(resume_skill_set) == 0 or len(job_skill_set) == 0:
            jaccard_similarity = 0.0
        else:
            intersection = resume_skill_set.intersection(job_skill_set)
            union = resume_skill_set.union(job_skill_set)
            jaccard_similarity = len(intersection) / len(union)
        
        # Calculate common skills count
        common_skills = resume_skill_set.intersection(job_skill_set)
        common_skills_count = len(common_skills)
        
        # Get embeddings for semantic similarity
        resume_embedding = sentence_transformer.encode([resume_text])[0]
        job_embedding = sentence_transformer.encode([job_description])[0]
        skill_vector_similarity = np.dot(resume_embedding, job_embedding) / (
            np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
        )
        
        # Calculate additional features
        skill_count_diff = abs(len(resume_skill_set) - len(job_skill_set))
        skill_count_ratio = common_skills_count / max(1, len(job_skill_set))
        skills_density = len(resume_skill_set) / max(1, len(resume_text.split()))
        
        # Prepare features for model prediction
        features = np.array([[
            jaccard_similarity,
            common_skills_count,
            skill_vector_similarity,
            skill_count_diff,
            skill_count_ratio,
            skills_density
        ]])
        
        # Scale features
        features_scaled = feature_scaler.transform(features)
        
        # Get match score prediction
        match_score = match_model.predict(features_scaled)[0]
        match_percentage = min(100, max(0, match_score * 100))
        
        # Calculate ATS score
        ats_score = min(100, max(0, (
            jaccard_similarity * 0.4 +
            skill_vector_similarity * 0.3 +
            skill_count_ratio * 0.3
        ) * 100))
        
        # Generate strengths and weaknesses
        strengths = [f"Strong skill: {skill}" for skill in list(common_skills)[:5]]
        weaknesses = [f"Missing skill: {skill}" for skill in list(job_skill_set - resume_skill_set)[:5]]
        
        # Generate recommendations
        recommendations = []
        if weaknesses:
            recommendations.append(f"Add missing skills: {', '.join(list(job_skill_set - resume_skill_set)[:3])}")
        if skills_density < 0.1:
            recommendations.append("Increase skills density in resume")
        if len(resume_text.split()) < 200:
            recommendations.append("Add more detailed content to resume")
        
        return {
            'match_score': round(match_percentage, 2),
            'ats_score': round(ats_score, 2),
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendations': recommendations
        }
        
    except Exception as e:
        print(f"Error in model-based analysis: {str(e)}")
        return _basic_match_analysis(resume_text, job_description)

def _basic_match_analysis(resume_text, job_description):
    """
    Final fallback method for basic match analysis.
    """
    return {
        'match_score': 50,
        'ats_score': 50,
        'strengths': ['Basic resume structure present'],
        'weaknesses': ['Unable to perform detailed analysis'],
        'recommendations': ['Please try again or contact support']
    } 