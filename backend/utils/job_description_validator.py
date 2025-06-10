import re
import logging
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobDescriptionValidator:
    def __init__(self):
        """Initialize the JobDescriptionValidator."""
        self.min_length = 100  # Minimum characters
        self.required_sections = [
            'responsibilities',
            'requirements',
            'qualifications'
        ]
        
    def validate_job_description(self, job_description: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate job description against set criteria.
        Returns: (is_valid, message, analysis)
        """
        try:
            # Check minimum length
            if len(job_description) < self.min_length:
                return False, "Job description is too short", {
                    'length': len(job_description),
                    'min_required': self.min_length
                }
            
            # Check for required sections
            missing_sections = []
            for section in self.required_sections:
                if not re.search(rf'\b{section}\b', job_description.lower()):
                    missing_sections.append(section)
            
            if missing_sections:
                return False, f"Missing required sections: {', '.join(missing_sections)}", {
                    'missing_sections': missing_sections
                }
            
            # Check for minimum content in each section
            sections_content = self._analyze_sections_content(job_description)
            if not sections_content['has_sufficient_content']:
                return False, "Sections lack sufficient detail", sections_content
            
            # Check for inappropriate content
            inappropriate_content = self._check_inappropriate_content(job_description)
            if inappropriate_content:
                return False, "Contains inappropriate content", {
                    'inappropriate_content': inappropriate_content
                }
            
            # All checks passed
            return True, "Job description is valid", {
                'length': len(job_description),
                'sections_analysis': sections_content,
                'has_inappropriate_content': False
            }
            
        except Exception as e:
            logger.error(f"Error validating job description: {str(e)}")
            return False, f"Error validating job description: {str(e)}", {}
    
    def _analyze_sections_content(self, job_description: str) -> Dict[str, Any]:
        """Analyze content of different sections."""
        try:
            sections = {
                'responsibilities': self._extract_section_content(job_description, 'responsibilities'),
                'requirements': self._extract_section_content(job_description, 'requirements'),
                'qualifications': self._extract_section_content(job_description, 'qualifications')
            }
            
            # Check if each section has sufficient content
            has_sufficient_content = all(
                len(content.split()) >= 20 for content in sections.values() if content
            )
            
            return {
                'sections': sections,
                'has_sufficient_content': has_sufficient_content,
                'word_counts': {
                    section: len(content.split()) for section, content in sections.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sections content: {str(e)}")
            return {
                'sections': {},
                'has_sufficient_content': False,
                'word_counts': {}
            }
    
    def _extract_section_content(self, text: str, section: str) -> str:
        """Extract content of a specific section."""
        try:
            # Look for section headers
            pattern = rf'\b{section}\b[:\s]+(.*?)(?=\b(?:responsibilities|requirements|qualifications|$)\b)'
            match = re.search(pattern, text.lower(), re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""
            
        except Exception as e:
            logger.error(f"Error extracting section content: {str(e)}")
            return ""
    
    def _check_inappropriate_content(self, text: str) -> list:
        """Check for inappropriate content in job description."""
        try:
            inappropriate_patterns = [
                r'\b(?:age|gender|race|religion|nationality|marital status)\b',
                r'\b(?:discriminate|discrimination)\b',
                r'\b(?:illegal|unlawful)\b',
                r'\b(?:unpaid|volunteer)\s+(?:work|position)\b',
                r'\b(?:commission\s+only)\b'
            ]
            
            found_issues = []
            for pattern in inappropriate_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    found_issues.append({
                        'pattern': pattern,
                        'context': text[max(0, match.start()-20):min(len(text), match.end()+20)]
                    })
            
            return found_issues
            
        except Exception as e:
            logger.error(f"Error checking inappropriate content: {str(e)}")
            return []

# Create a singleton instance
job_description_validator = JobDescriptionValidator()

def validate_job_description(job_description: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate job description.
    Returns: (is_valid, message, analysis)
    """
    return job_description_validator.validate_job_description(job_description) 