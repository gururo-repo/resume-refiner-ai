import requests
import os
from typing import Dict, List, Optional
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

    def _create_improvement_prompt(self, resume_text: str, jd_text: str) -> str:
        """Create a detailed prompt for the LLM."""
        return f"""As an expert resume writer and career coach, improve this resume to better match the job description.
Focus on:
1. Quantifying achievements with metrics and numbers
2. Highlighting relevant skills and experiences
3. Using strong action verbs
4. Improving clarity and impact
5. Tailoring content to the job requirements

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

Improved Resume:"""

    def _extract_key_requirements(self, jd_text: str) -> List[str]:
        """Extract key requirements from job description."""
        # Use TF-IDF to find important terms
        tfidf_matrix = self.tfidf.fit_transform([jd_text])
        feature_names = self.tfidf.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        top_indices = scores.argsort()[-10:][::-1]
        return [feature_names[i] for i in top_indices]

    def _ml_based_improvement(self, resume_text: str, jd_text: str) -> str:
        """ML-based fallback for resume improvement."""
        # Extract key requirements
        key_requirements = self._extract_key_requirements(jd_text)
        
        # Get embeddings
        resume_embedding = self.model.encode([resume_text])[0]
        jd_embedding = self.model.encode([jd_text])[0]
        
        # Calculate similarity
        similarity = np.dot(resume_embedding, jd_embedding) / (
            np.linalg.norm(resume_embedding) * np.linalg.norm(jd_embedding)
        )
        
        # Generate improvement suggestions
        suggestions = []
        if similarity < 0.7:
            suggestions.append("Add more relevant keywords from the job description")
        
        # Check for quantified achievements
        if not re.search(r'\d+%|\$\d+|\d+\s*(?:million|billion|thousand)', resume_text):
            suggestions.append("Add quantifiable achievements and metrics")
        
        # Check for action verbs
        action_verbs = ['achieved', 'improved', 'increased', 'developed', 'implemented', 'led']
        if not any(verb in resume_text.lower() for verb in action_verbs):
            suggestions.append("Use more strong action verbs")
        
        # Format suggestions
        improved_resume = resume_text + "\n\nSuggested Improvements:\n"
        for suggestion in suggestions:
            improved_resume += f"- {suggestion}\n"
        
        return improved_resume

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

    def improve_resume(self, resume_text: str, jd_text: str) -> Dict:
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

def suggest_resume(resume_text: str, jd_text: str) -> Dict:
    """Main function to get resume suggestions."""
    improver = ResumeImprover()
    return improver.improve_resume(resume_text, jd_text) 