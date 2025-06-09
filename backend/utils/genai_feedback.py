import logging
from typing import Dict, Any
import google.generativeai as genai
import os
from dotenv import load_dotenv
from .groq_analyzer import get_groq_analysis

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class GenAIFeedback:
    def __init__(self):
        """Initialize the GenAI feedback generator."""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            logger.warning("Google API key not found. GenAI feedback will be disabled.")
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
        except Exception as e:
            logger.error(f"Failed to initialize GenAI model: {str(e)}")
            self.model = None
    
    def generate_feedback(self, resume_text: str, job_description: str = None) -> Dict[str, Any]:
        """
        Generate feedback using Groq API first, then fallback to default feedback.
        """
        try:
            # Try Groq API first
            try:
                groq_analysis = get_groq_analysis(resume_text, job_description)
                if groq_analysis:
                    logger.info("Successfully got feedback from Groq API")
                    return {
                        'strengths': groq_analysis.get('strengths', []),
                        'weaknesses': groq_analysis.get('weaknesses', []),
                        'improvements': groq_analysis.get('improvement_tips', []),
                        'format_analysis': groq_analysis.get('format_analysis', {}),
                        'skills_analysis': groq_analysis.get('skills_analysis', {})
                    }
            except Exception as e:
                logger.warning(f"Groq feedback generation failed: {str(e)}")
            
            logger.info("Using default feedback as fallback")
            return self._get_default_feedback()
            
        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}")
            return self._get_default_feedback()

    def _get_default_feedback(self) -> Dict[str, Any]:
        """Return default feedback when all methods fail."""
        return {
            'strengths': [],
            'weaknesses': [],
            'improvements': ['Unable to generate feedback'],
            'format_analysis': {
                'score': 0,
                'issues': [],
                'suggestions': []
            },
            'skills_analysis': {
                'matching_skills': [],
                'missing_skills': [],
                'skill_gaps': []
            }
        }
    
    def get_feedback(self, resume_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """
        Get AI-powered feedback on the resume.
        
        Args:
            resume_data: Parsed resume data
            job_description: Job description text
            
        Returns:
            Dictionary containing AI feedback
        """
        try:
            # Prepare the prompt
            prompt = self._prepare_prompt(resume_data, job_description)
            
            # Get response from the model
            response = self.model.generate_content(prompt)
            
            # Parse the response
            feedback = self._parse_response(response.text)
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error getting GenAI feedback: {str(e)}")
            raise
    
    def _prepare_prompt(self, resume_data: Dict[str, Any], job_description: str) -> str:
        """Prepare the prompt for the GenAI model."""
        prompt = f"""Analyze this resume against the job description and provide detailed feedback.
        
Job Description:
{job_description}

Resume Content:
{resume_data['raw_text']}

Please provide feedback in the following areas:
1. Strengths: What are the candidate's key strengths that match the job requirements?
2. Weaknesses: What areas need improvement to better match the job requirements?
3. Tips: Specific suggestions for improving the resume
4. Rewritten Sections: Provide improved versions of key sections

Format your response as a JSON object with the following structure:
{{
    "strengths": ["strength1", "strength2", ...],
    "weaknesses": ["weakness1", "weakness2", ...],
    "tips": ["tip1", "tip2", ...],
    "rewritten_sections": {{
        "summary": "improved summary",
        "experience": "improved experience section",
        "skills": "improved skills section"
    }}
}}

Focus on:
- ATS optimization
- Quantifiable achievements
- Relevant skills and experience
- Professional language and formatting
"""
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the GenAI response into a structured format."""
        try:
            # Extract JSON from response
            import json
            import re
            
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                feedback = json.loads(json_match.group())
            else:
                # Fallback to basic parsing if JSON not found
                feedback = {
                    "strengths": [],
                    "weaknesses": [],
                    "tips": [],
                    "rewritten_sections": {
                        "summary": "",
                        "experience": "",
                        "skills": ""
                    }
                }
                
                # Parse sections
                sections = response_text.split('\n\n')
                for section in sections:
                    if section.lower().startswith('strengths'):
                        feedback['strengths'] = self._extract_list_items(section)
                    elif section.lower().startswith('weaknesses'):
                        feedback['weaknesses'] = self._extract_list_items(section)
                    elif section.lower().startswith('tips'):
                        feedback['tips'] = self._extract_list_items(section)
                    elif section.lower().startswith('rewritten'):
                        feedback['rewritten_sections'] = self._extract_rewritten_sections(section)
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error parsing GenAI response: {str(e)}")
            return {
                "strengths": ["Error parsing feedback"],
                "weaknesses": [],
                "tips": [],
                "rewritten_sections": {
                    "summary": "",
                    "experience": "",
                    "skills": ""
                }
            }
    
    def _extract_list_items(self, text: str) -> list:
        """Extract list items from text."""
        items = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.lower().startswith(('strengths', 'weaknesses', 'tips')):
                # Remove bullet points or numbers
                line = re.sub(r'^[\d\.\-\*]+', '', line).strip()
                if line:
                    items.append(line)
        return items
    
    def _extract_rewritten_sections(self, text: str) -> Dict[str, str]:
        """Extract rewritten sections from text."""
        sections = {
            "summary": "",
            "experience": "",
            "skills": ""
        }
        
        current_section = None
        section_content = []
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.lower().startswith(('summary', 'experience', 'skills')):
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content)
                current_section = line.lower().split(':')[0]
                section_content = []
            elif current_section:
                section_content.append(line)
        
        # Add the last section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return sections

# Initialize global feedback instance
genai_feedback = GenAIFeedback()

def get_genai_feedback(resume_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
    """
    Get AI-powered feedback on the resume.
    
    Args:
        resume_data: Parsed resume data
        job_description: Job description text
        
    Returns:
        Dictionary containing AI feedback
    """
    return genai_feedback.get_feedback(resume_data, job_description) 