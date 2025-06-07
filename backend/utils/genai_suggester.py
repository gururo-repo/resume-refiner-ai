import requests
import os
from typing import Dict, List, Optional, Any
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class ResumeImprover:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.llm_api_key = os.getenv('GROK_API_KEY')
        self.llm_api_url = os.getenv('GROK_API_URL', 'https://api.grok.ai/v1/chat/completions')

    def _create_ats_prompt(self, resume_text: str, jd_text: str) -> str:
        """Create a detailed prompt for ATS scoring."""
        return f"""As an expert ATS (Applicant Tracking System) analyzer, evaluate this resume against the job description.
Provide a detailed analysis including:

1. ATS Compatibility Score (0-100)
2. Key Strengths:
   - Matching skills and keywords
   - Relevant experience
   - Quantified achievements
3. Areas for Improvement:
   - Missing keywords
   - Formatting issues
   - Content gaps
4. Specific Recommendations for ATS optimization

Job Description:
{jd_text}

Resume:
{resume_text}

Please provide your analysis in a structured format with the following sections:
- ATS Score (0-100)
- List of Strengths
- List of Weaknesses
- List of Recommendations
- List of Matched Keywords
- List of Missing Keywords"""

    def _create_improvement_prompt(self, resume_text: str, jd_text: str) -> str:
        """Create a detailed prompt for resume improvement."""
        return f"""As an expert resume writer and career coach, improve this resume to better match the job description.
Focus on:
1. Quantifying achievements with metrics and numbers
2. Highlighting relevant skills and experiences
3. Using strong action verbs
4. Improving clarity and impact
5. Tailoring content to the job requirements
6. Optimizing for ATS systems

Job Description:
{jd_text}

Current Resume:
{resume_text}

Please provide an improved version of the resume that:
- Maintains the same structure but enhances content
- Quantifies achievements where possible
- Uses industry-standard terminology
- Highlights relevant skills and experiences
- Is more impactful and ATS-friendly
- Includes all important keywords from the job description

Improved Resume:"""

    def _extract_key_requirements(self, jd_text: str) -> List[str]:
        """Extract key requirements from job description."""
        # Use TF-IDF to find important terms
        tfidf_matrix = self.tfidf.fit_transform([jd_text])
        feature_names = self.tfidf.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        top_indices = scores.argsort()[-10:][::-1]
        return [feature_names[i] for i in top_indices]

    def _ml_based_ats_score(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """ML-based fallback for ATS scoring."""
        # Extract key requirements
        key_requirements = self._extract_key_requirements(jd_text)
        
        # Get embeddings
        resume_embedding = self.model.encode([resume_text])[0]
        jd_embedding = self.model.encode([jd_text])[0]
        
        # Calculate similarity
        similarity = np.dot(resume_embedding, jd_embedding) / (
            np.linalg.norm(resume_embedding) * np.linalg.norm(jd_embedding)
        )
        
        # Basic ATS score calculation
        ats_score = min(100, max(0, similarity * 100))
        
        # Find matching and missing keywords
        resume_lower = resume_text.lower()
        matched_keywords = [kw for kw in key_requirements if kw.lower() in resume_lower]
        missing_keywords = [kw for kw in key_requirements if kw.lower() not in resume_lower]
        
        return {
            'ats_score': round(ats_score, 2),
            'strengths': [
                f"Matches {len(matched_keywords)} key requirements",
                "Good semantic similarity with job description" if similarity > 0.7 else None
            ],
            'weaknesses': [
                f"Missing {len(missing_keywords)} key requirements",
                "Low semantic similarity with job description" if similarity < 0.5 else None
            ],
            'recommendations': [
                "Add missing keywords naturally in the content",
                "Improve keyword density",
                "Use more industry-standard terminology"
            ],
            'keyword_matches': matched_keywords,
            'missing_keywords': missing_keywords
        }

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            # First try parsing as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # If not JSON, try to extract structured information
            sections = {
                'ats_score': None,
                'strengths': [],
                'weaknesses': [],
                'recommendations': [],
                'keyword_matches': [],
                'missing_keywords': []
            }
            
            # Extract ATS score
            score_match = re.search(r'ATS Score:\s*(\d+)', response)
            if score_match:
                sections['ats_score'] = int(score_match.group(1))
            
            # Extract sections using regex patterns
            section_patterns = {
                'strengths': r'Strengths:?\s*(.*?)(?=\n\s*(?:Weaknesses|Recommendations|$))',
                'weaknesses': r'Weaknesses:?\s*(.*?)(?=\n\s*(?:Recommendations|$))',
                'recommendations': r'Recommendations:?\s*(.*?)(?=\n\s*(?:Keywords|$))',
                'keyword_matches': r'Matched Keywords:?\s*(.*?)(?=\n\s*(?:Missing Keywords|$))',
                'missing_keywords': r'Missing Keywords:?\s*(.*?)(?=\n\s*$)'
            }
            
            for key, pattern in section_patterns.items():
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    items = [item.strip() for item in match.group(1).split('\n') if item.strip()]
                    sections[key] = items
            
            return sections

    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """Call the LLM API with error handling."""
        try:
            headers = {
                'Authorization': f'Bearer {self.llm_api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': 'grok-1',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.7,
                'max_tokens': 2000
            }
            response = requests.post(self.llm_api_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"LLM API error: {str(e)}")
            return None

    def analyze_ats_compatibility(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Analyze ATS compatibility using LLM or fallback to ML."""
        # Try LLM first
        prompt = self._create_ats_prompt(resume_text, jd_text)
        llm_analysis = self._call_llm_api(prompt)
        
        if llm_analysis:
            parsed_analysis = self._parse_llm_response(llm_analysis)
            return {
                'analysis': parsed_analysis,
                'method': 'llm',
                'confidence': 'high' if isinstance(parsed_analysis, dict) else 'medium'
            }
        
        # Fallback to ML-based analysis
        ml_analysis = self._ml_based_ats_score(resume_text, jd_text)
        return {
            'analysis': ml_analysis,
            'method': 'ml',
            'confidence': 'low'
        }

    def improve_resume(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Main function to improve resume with LLM or fallback to ML."""
        # Try LLM first
        prompt = self._create_improvement_prompt(resume_text, jd_text)
        llm_suggestion = self._call_llm_api(prompt)
        
        if llm_suggestion:
            return {
                'improved_resume': llm_suggestion,
                'method': 'llm',
                'confidence': 'high'
            }
        
        # Fallback to ML-based improvement
        ml_suggestion = self._ml_based_improvement(resume_text, jd_text)
        return {
            'improved_resume': ml_suggestion,
            'method': 'ml',
            'confidence': 'medium'
        }

def analyze_resume(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """Main function to analyze resume and get suggestions."""
    improver = ResumeImprover()
    
    # Get ATS analysis
    ats_analysis = improver.analyze_ats_compatibility(resume_text, jd_text)
    
    # Get improvement suggestions
    improvement = improver.improve_resume(resume_text, jd_text)
    
    # Extract the analysis from the response
    analysis = ats_analysis.get('analysis', {})
    
    return {
        'ats_score': analysis.get('ats_score', 0),
        'match_score': analysis.get('ats_score', 0),  # Using ATS score as match score for now
        'strengths': analysis.get('strengths', []),
        'weaknesses': analysis.get('weaknesses', []),
        'recommendations': analysis.get('recommendations', [])
    } 