import os
import json
import logging
import requests
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .model_loader import get_models, get_match_model, get_feature_scaler, get_model_features, get_sentence_transformer
from .skill_matcher import skill_matcher, extract_skills
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ResumeImprover:
    def __init__(self):
        """Initialize the GenAI suggester."""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            logger.warning("Google API key not found. GenAI suggestions will be disabled.")
            self.model = None
            return
        
        try:
            genai.configure(api_key=self.api_key)
            model = "models/gemini-1.5-pro"
            url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
            
            self.model = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 2048,
                }
            )
            logger.info(f"Successfully initialized GenAI model: {model}")
            
            # Initialize ML models as fallback
            self.models = get_models()
            self.match_model = get_match_model()
            self.feature_scaler = get_feature_scaler()
            self.model_features = get_model_features()
            self.sentence_transformer = get_sentence_transformer()
            
            # Test API connection
            self._test_api_connection()
            
            logger.info("Successfully initialized ResumeImprover with Google AI API (Flash model)")
        except Exception as e:
            logger.error(f"Failed to initialize ResumeImprover: {str(e)}")
            raise

    def _test_api_connection(self):
        """Test the Google AI API connection."""
        try:
            test_prompt = "Test connection"
            response = self.model.generate_content(test_prompt)
            
            if not response.text:
                raise ValueError("No response from Google AI API")
                
            logger.info("Successfully tested Google AI API connection")
        except Exception as e:
            logger.error(f"Failed to connect to Google AI API: {str(e)}")
            raise

    def _call_ai_api(self, prompt, temperature=0.2, max_retries=3):
        """Make a call to the Google AI API with improved error handling and retries."""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=1024,  # Reduced for Flash model
                        top_p=0.95
                    )
                )
                
                if not response.text:
                    raise ValueError("No response from Google AI API")
                
                # Try to parse the response as JSON
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    # If response is not valid JSON, try to extract JSON from the text
                    json_start = response.text.find('{')
                    json_end = response.text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response.text[json_start:json_end]
                        return json.loads(json_str)
                    raise ValueError("Response is not in valid JSON format")
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to call Google AI API after {max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                continue

    def analyze_ats_compatibility(self, resume_text, job_description):
        """
        Analyze resume compatibility with job description using Google AI API first,
        falling back to ML models if API fails.
        """
        try:
            prompt = f"""You are an ATS (Applicant Tracking System) expert specializing in resume optimization.

IMPORTANT: Return the analysis in the following strict JSON format WITHOUT ANY ADDITIONAL TEXT

Analyze this resume for ATS compatibility:
{resume_text}

Job Description:
{job_description}

Return the analysis in the following strict JSON format without any additional text:
{{
    "overall": number between 0-100,
    "keywords": array of strings containing detected keywords,
    "missing_keywords": array of strings containing important missing keywords,
    "format_score": number between 0-100,
    "suggestions": array of strings containing specific improvement suggestions
}}"""

            try:
                analysis = self._call_ai_api(prompt)
                logger.info("Successfully completed ATS compatibility analysis using Google AI API")
                return analysis
            except Exception as e:
                logger.warning(f"Google AI API failed, falling back to ML models: {str(e)}")
                return self._fallback_ats_analysis(resume_text, job_description)
            
        except Exception as e:
            logger.error(f"Failed to analyze ATS compatibility: {str(e)}")
            return self._get_default_ats_analysis()

    def analyze_skills(self, resume_text, job_description):
        """
        Analyze skills in resume against job description using Google AI API first,
        falling back to ML models if API fails.
        """
        try:
            prompt = f"""You are an experienced technical recruiter and job matching specialist.

IMPORTANT: Return the analysis in the following strict JSON format WITHOUT ANY ADDITIONAL TEXT

Compare this resume with the job description and analyze the match:

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Return the analysis in the following strict JSON format without any additional text:
{{
    "score": number between 0-100 representing overall match percentage,
    "matching_skills": array of strings containing skills that match the job requirements,
    "missing_skills": array of strings containing required skills that are missing,
    "recommendations": array of strings containing specific suggestions for improvement,
    "relevance": number between 0-100 representing experience relevance,
    "skill_gaps": array of objects containing {{"skill": string, "importance": number between 0-100, "suggestion": string}}
}}"""

            try:
                analysis = self._call_ai_api(prompt)
                logger.info("Successfully completed skills analysis using Google AI API")
                return analysis
            except Exception as e:
                logger.warning(f"Google AI API failed, falling back to ML models: {str(e)}")
                return skill_matcher.extract_and_match_skills(resume_text, job_description)
            
        except Exception as e:
            logger.error(f"Failed to analyze skills: {str(e)}")
            return self._get_default_skills_analysis()

    def analyze_resume_structure(self, resume_text):
        """
        Analyze resume structure using Google AI API first,
        falling back to ML models if API fails.
        """
        try:
            prompt = f"""You are an expert resume structure analyzer focused on format and organization.

IMPORTANT: Return the analysis in the following strict JSON format WITHOUT ANY ADDITIONAL TEXT

Analyze the structure and formatting of this resume:
{resume_text}

Return the analysis in the following strict JSON format without any additional text:
{{
    "completeness": number between 0-100 representing how complete the resume is,
    "sections_present": array of strings containing detected resume sections,
    "sections_missing": array of strings containing important missing sections,
    "suggestions": array of strings containing formatting and structure improvements,
    "readability": number between 0-100 representing how readable the resume is,
    "format_issues": array of objects containing {{"issue": string, "severity": "high|medium|low", "suggestion": string}}
}}"""

            try:
                analysis = self._call_ai_api(prompt)
                logger.info("Successfully completed resume structure analysis using Google AI API")
                return analysis
            except Exception as e:
                logger.warning(f"Google AI API failed, falling back to ML models: {str(e)}")
                return self._fallback_structure_analysis(resume_text)
            
        except Exception as e:
            logger.error(f"Failed to analyze resume structure: {str(e)}")
            return self._get_default_structure_analysis()

    def analyze_roles(self, resume_text, job_description):
        """
        Predict potential roles based on resume and job description using Google AI API first,
        falling back to ML models if API fails.
        """
        try:
            prompt = f"""You are an experienced technical recruiter and role prediction specialist.

IMPORTANT: Return the analysis in the following strict JSON format WITHOUT ANY ADDITIONAL TEXT

Analyze this resume and job description to predict potential roles:

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Return the analysis in the following strict JSON format without any additional text:
{{
    "predicted_roles": [
        {{
            "role": string,
            "confidence": number between 0-100,
            "reasoning": string,
            "required_skills": array of strings,
            "matching_skills": array of strings
        }}
    ],
    "role_match": {{
        "primary_role": string,
        "match_confidence": number between 0-100,
        "key_qualifications": [string],
        "missing_qualifications": [string],
        "career_path": array of objects containing {{"role": string, "years_experience": number, "required_skills": [string]}}
    }}
}}"""

            try:
                analysis = self._call_ai_api(prompt)
                logger.info("Successfully completed role prediction using Google AI API")
                return analysis
            except Exception as e:
                logger.warning(f"Google AI API failed, falling back to ML models: {str(e)}")
                return self._fallback_role_analysis(resume_text, job_description)
            
        except Exception as e:
            logger.error(f"Failed to predict roles: {str(e)}")
            return self._get_default_role_analysis()

    def _fallback_ats_analysis(self, resume_text, job_description):
        """Fallback ATS analysis using ML models."""
        try:
            resume_embedding = self.sentence_transformer.encode([resume_text])[0]
            job_embedding = self.sentence_transformer.encode([job_description])[0]
            
            similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
            ats_score = int(similarity * 100)
            
            # Extract keywords using skill matcher
            skills = extract_skills(resume_text)
            keywords = []
            for category, skill_list in skills.items():
                keywords.extend(skill_list)
            
            return {
                "overall": ats_score,
                "keywords": keywords,
                "missing_keywords": [],
                "format_score": 0,
                "suggestions": ["Using fallback ML model analysis"]
            }
        except Exception as e:
            logger.error(f"Fallback ATS analysis failed: {str(e)}")
            return self._get_default_ats_analysis()

    def _get_default_ats_analysis(self):
        """Return default ATS analysis when all methods fail."""
        return {
            "overall": 0,
            "keywords": [],
            "missing_keywords": [],
            "format_score": 0,
            "suggestions": ["Unable to perform ATS analysis"]
        }

    def _get_default_skills_analysis(self):
        """Return default skills analysis when all methods fail."""
        return {
            "score": 0,
            "matching_skills": [],
            "missing_skills": [],
            "recommendations": ["Unable to perform skills analysis"],
            "relevance": 0,
            "skill_gaps": []
        }

    def _get_default_structure_analysis(self):
        """Return default structure analysis when all methods fail."""
        return {
            "completeness": 0,
            "sections_present": [],
            "sections_missing": [],
            "suggestions": ["Unable to perform structure analysis"],
            "readability": 0,
            "format_issues": []
        }

    def _get_default_role_analysis(self):
        """Return default role analysis when all methods fail."""
        return {
            "predicted_roles": [],
            "role_match": {
                "primary_role": "Unknown",
                "match_confidence": 0,
                "key_qualifications": [],
                "missing_qualifications": [],
                "career_path": []
            }
        }

def get_resume_improver():
    """Get or create a ResumeImprover instance."""
    return ResumeImprover() 