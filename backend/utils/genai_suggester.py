import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from .model_loader import get_models, get_match_model, get_feature_scaler, get_model_features, get_sentence_transformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Google Generative AI
try:
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    logger.info("Successfully configured Google Generative AI")
except Exception as e:
    logger.error(f"Failed to configure Google Generative AI: {str(e)}")
    raise

class ResumeImprover:
    def __init__(self):
        try:
            self.models = get_models()
            self.match_model = get_match_model()
            self.feature_scaler = get_feature_scaler()
            self.model_features = get_model_features()
            self.sentence_transformer = get_sentence_transformer()
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Successfully initialized ResumeImprover")
        except Exception as e:
            logger.error(f"Failed to initialize ResumeImprover: {str(e)}")
            raise

    def analyze_ats_compatibility(self, resume_text, job_description):
        """
        Analyze resume compatibility with job description using LLM.
        """
        try:
            prompt = f"""
            Analyze the following resume against the job description and provide:
            1. ATS compatibility score (0-100)
            2. Match score (0-100)
            3. Key strengths
            4. Areas for improvement
            5. Specific recommendations

            Resume:
            {resume_text}

            Job Description:
            {job_description}

            Provide the response in JSON format with the following structure:
            {{
                "ats_score": <score>,
                "match_score": <score>,
                "strengths": [<list of strengths>],
                "weaknesses": [<list of weaknesses>],
                "recommendations": [<list of recommendations>]
            }}
            """

            response = self.model.generate_content(prompt)
            analysis = json.loads(response.text)
            
            logger.info("Successfully completed ATS compatibility analysis")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze ATS compatibility: {str(e)}")
            raise

    def analyze_skills(self, resume_text, job_description):
        """
        Analyze skills in resume against job description using LLM.
        """
        try:
            prompt = f"""
            Analyze the skills in the following resume against the job description and provide:
            1. Matched skills
            2. Missing skills
            3. Additional skills

            Resume:
            {resume_text}

            Job Description:
            {job_description}

            Provide the response in JSON format with the following structure:
            {{
                "matched_skills": [<list of matched skills>],
                "missing_skills": [<list of missing skills>],
                "additional_skills": [<list of additional skills>]
            }}
            """

            response = self.model.generate_content(prompt)
            analysis = json.loads(response.text)
            
            logger.info("Successfully completed skills analysis")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze skills: {str(e)}")
            raise

    def analyze_roles(self, resume_text, job_description):
        """
        Predict potential roles based on resume and job description using LLM.
        """
        try:
            prompt = f"""
            Analyze the following resume and job description to predict potential roles and provide:
            1. Top 3 predicted roles
            2. Confidence scores for each role

            Resume:
            {resume_text}

            Job Description:
            {job_description}

            Provide the response in JSON format with the following structure:
            {{
                "predicted_roles": [<list of top 3 roles>],
                "confidence_scores": [<list of confidence scores>]
            }}
            """

            response = self.model.generate_content(prompt)
            analysis = json.loads(response.text)
            
            logger.info("Successfully completed role prediction")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to predict roles: {str(e)}")
            raise

# Create a singleton instance
resume_improver = ResumeImprover()

def get_resume_improver():
    """
    Get the singleton instance of ResumeImprover.
    This is the main entry point for resume improvement functionality.
    """
    return resume_improver 