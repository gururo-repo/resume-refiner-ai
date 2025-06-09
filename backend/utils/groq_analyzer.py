import os
import json
import logging
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from .ml_fallback import get_ml_fallback
from .resume_parser import ResumeParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GroqAnalyzer:
    def __init__(self):
        """Initialize the Groq analyzer."""
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            logger.warning("Groq API key not found. Groq analysis will be disabled.")
            self.enabled = False
            return
        
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.enabled = True
        self.supported_models = [
            "llama3-70b-8192",
            "mistral-7b-instruct",
            "gemma-7b-it"
        ]
        # In-memory cache for consistent responses
        self._analysis_cache = {}
        # Initialize resume parser
        self.resume_parser = ResumeParser()
        logger.info("Successfully initialized Groq analyzer")

    def _get_cache_key(self, resume_text: str, job_description: str = None) -> str:
        """Generate a cache key based on resume and job description content."""
        content = f"{resume_text}{job_description if job_description else ''}"
        return content

    def _is_valid_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Check if the analysis has valid scores and non-empty content."""
        try:
            # Check if analysis is None or empty
            if not analysis:
                return False

            # Check if scores are valid numbers greater than 0
            ats_score = analysis.get('ats_score', 0)
            job_match_score = analysis.get('job_match_score', 0)
            
            if not isinstance(ats_score, (int, float)) or ats_score <= 0:
                logger.warning(f"Invalid ATS score: {ats_score}")
                return False
            if not isinstance(job_match_score, (int, float)) or job_match_score <= 0:
                logger.warning(f"Invalid job match score: {job_match_score}")
                return False

            # Check if required fields have non-empty content
            required_fields = {
                'strengths': list,
                'weaknesses': list,
                'improvement_tips': list,
                'skills_analysis': dict,
                'format_analysis': dict
            }

            for field, field_type in required_fields.items():
                if field not in analysis:
                    logger.warning(f"Missing required field: {field}")
                    return False
                if not isinstance(analysis[field], field_type):
                    logger.warning(f"Invalid type for field {field}: {type(analysis[field])}")
                    return False
                if field_type == list and not analysis[field]:
                    logger.warning(f"Empty list for field: {field}")
                    return False
                if field_type == dict and not analysis[field]:
                    logger.warning(f"Empty dict for field: {field}")
                    return False

            # Check skills analysis content
            skills_analysis = analysis.get('skills_analysis', {})
            if not skills_analysis.get('matching_skills') and not skills_analysis.get('missing_skills'):
                logger.warning("Empty skills analysis")
                return False

            # Check format analysis content
            format_analysis = analysis.get('format_analysis', {})
            format_score = format_analysis.get('score', 0)
            if not isinstance(format_score, (int, float)) or format_score <= 0:
                logger.warning(f"Invalid format score: {format_score}")
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating analysis: {str(e)}")
            return False

    def analyze_resume(self, resume_text: str, job_description: str = None) -> Dict[str, Any]:
        """
        Analyze resume using Groq API.
        
        Args:
            resume_text: The resume text to analyze
            job_description: Optional job description for matching analysis
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.enabled:
            logger.warning("Groq analysis is disabled due to missing API key")
            return self._get_fallback_analysis(resume_text, job_description)

        if not resume_text:
            logger.error("Empty resume text provided")
            return self._get_fallback_analysis(resume_text, job_description)

        # Check in-memory cache first
        cache_key = self._get_cache_key(resume_text, job_description)
        cached_analysis = self._analysis_cache.get(cache_key)
        if cached_analysis and self._is_valid_analysis(cached_analysis):
            logger.info("Retrieved valid analysis from in-memory cache")
            return cached_analysis

        logger.info(f"Starting Groq analysis with resume length: {len(resume_text)}")
        if job_description:
            logger.info(f"Job description length: {len(job_description)}")

        # Try each supported model in order
        for model in self.supported_models:
            try:
                logger.info(f"Attempting analysis with model: {model}")
                analysis = self._try_model_analysis(model, resume_text, job_description)
                if analysis and self._is_valid_analysis(analysis):
                    logger.info(f"Successfully completed Groq resume analysis using {model}")
                    logger.debug(f"Analysis result: {json.dumps(analysis, indent=2)}")
                    # Store valid analysis in cache
                    self._analysis_cache[cache_key] = analysis
                    return analysis
                else:
                    logger.warning(f"Model {model} returned invalid or empty analysis")
            except Exception as e:
                logger.warning(f"Failed to analyze with model {model}: {str(e)}")
                continue

        logger.error("All Groq models failed or returned empty analysis, falling back to trained models")
        return self._get_fallback_analysis(resume_text, job_description)

    def _try_model_analysis(self, model: str, resume_text: str, job_description: str = None) -> Optional[Dict[str, Any]]:
        """Try analysis with a specific model."""
        try:
            # Prepare the prompt
            prompt = self._prepare_prompt(resume_text, job_description)
            logger.debug(f"Prepared prompt for model {model}")
            
            # Make API request
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You're a professional career coach and ATS resume analyzer. Provide detailed analysis in JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 2048
            }

            logger.info(f"Sending request to Groq API with model {model}")
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            # Handle specific error cases
            if response.status_code == 400:
                error_data = response.json()
                if "model" in error_data.get("error", {}).get("message", "").lower():
                    logger.warning(f"Model {model} not available")
                    return None
                else:
                    logger.error(f"Bad request error: {error_data}")
                    return None
            elif response.status_code == 429:
                logger.error("Rate limit exceeded for Groq API")
                return None
                
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            logger.debug(f"Raw API response: {json.dumps(result, indent=2)}")
            
            analysis = self._parse_response(result["choices"][0]["message"]["content"])
            
            if analysis:
                analysis['model_used'] = model
                analysis['analysis_source'] = 'groq'
                logger.info(f"Successfully parsed analysis from model {model}")
                return analysis
            else:
                logger.warning(f"Failed to parse response from model {model}")
                return None
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while using model {model}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error with model {model}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error with model {model}: {str(e)}")
            return None

    def _prepare_prompt(self, resume_text: str, job_description: str = None) -> str:
        """Prepare the analysis prompt."""
        job_desc_part = f"\nJob Description:\n{job_description}" if job_description else ""
        
        prompt = f"""Analyze this resume and provide detailed feedback in JSON format.

Resume:
{resume_text}{job_desc_part}

Return the analysis in the following JSON format:
{{
    "ats_score": number between 0-100,
    "job_match_score": number between 0-100,
    "strengths": [string],
    "weaknesses": [string],
    "improvement_tips": [string],
    "skills_analysis": {{
        "matching_skills": [string],
        "missing_skills": [string],
        "skill_gaps": [string]
    }},
    "format_analysis": {{
        "score": number between 0-100,
        "issues": [string],
        "suggestions": [string]
    }},
    "role_match": {{
        "primary_role": string,
        "match_confidence": number between 0-100
    }}
}}

Focus on:
- ATS optimization
- Skills matching
- Format and structure
- Quantifiable achievements
- Professional language"""

        return prompt

    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse the Groq API response."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                # Validate required fields
                required_fields = ['ats_score', 'job_match_score', 'strengths', 'weaknesses']
                if all(field in analysis for field in required_fields):
                    logger.info("Successfully parsed and validated response")
                    return analysis
                else:
                    missing_fields = [field for field in required_fields if field not in analysis]
                    logger.warning(f"Response missing required fields: {missing_fields}")
                    return None
            
            logger.warning("Could not parse Groq response as JSON")
            logger.debug(f"Raw response text: {response_text}")
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Groq response: {str(e)}")
            logger.debug(f"Failed to parse text: {response_text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {str(e)}")
            return None

    def _get_fallback_analysis(self, resume_text: str, job_description: str = None) -> Dict[str, Any]:
        """Get analysis using trained models when Groq API fails."""
        try:
            # Parse resume first
            resume_data = self.resume_parser.parse(resume_text)
            
            # Get ML-based analysis
            ml_analysis = get_ml_fallback(resume_data, job_description)
            
            # Format the analysis to match the expected structure
            analysis = {
                "ats_score": ml_analysis.get('ats_score', 0),
                "job_match_score": ml_analysis.get('job_match_score', 0),
                "strengths": ml_analysis.get('strengths', []),
                "weaknesses": ml_analysis.get('weaknesses', []),
                "improvement_tips": ml_analysis.get('tips', []),
                "skills_analysis": {
                    "matching_skills": ml_analysis.get('skills_analysis', {}).get('matching_skills', []),
                    "missing_skills": ml_analysis.get('skills_analysis', {}).get('missing_skills', []),
                    "skill_gaps": ml_analysis.get('skills_analysis', {}).get('skill_gaps', [])
                },
                "format_analysis": {
                    "score": ml_analysis.get('format_analysis', {}).get('score', 0),
                    "issues": ml_analysis.get('format_analysis', {}).get('issues', []),
                    "suggestions": ml_analysis.get('format_analysis', {}).get('suggestions', [])
                },
                "role_match": {
                    "primary_role": ml_analysis.get('role_match', {}).get('primary_role', 'Unknown'),
                    "match_confidence": ml_analysis.get('role_match', {}).get('match_confidence', 0)
                },
                "analysis_source": "trained_models"
            }
            
            logger.info("Successfully completed fallback analysis using trained models")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in fallback analysis: {str(e)}")
            return {
                "ats_score": 0,
                "job_match_score": 0,
                "format_analysis": {"score": 0},
                "analysis_source": "fallback"
            }

# Initialize global analyzer instance
groq_analyzer = GroqAnalyzer()

def get_groq_analysis(resume_text: str, job_description: str = None) -> Optional[Dict[str, Any]]:
    """
    Get resume analysis from Groq API.
    
    Args:
        resume_text: The text content of the resume
        job_description: The job description text
        
    Returns:
        Optional[Dict[str, Any]]: Analysis results or None if analysis fails
    """
    return groq_analyzer.analyze_resume(resume_text, job_description) 