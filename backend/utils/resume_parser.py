import PyPDF2
import docx
import io
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeParser:
    def __init__(self):
        """Initialize the ResumeParser."""
        logger.info("Successfully initialized ResumeParser")
    
    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """
        Parse resume from PDF or DOCX file and extract structured information.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Dictionary containing parsed resume data
        """
        try:
            file_extension = file_path.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                text = self._extract_text_from_pdf(file_path)
            elif file_extension in ['docx', 'doc']:
                text = self._extract_text_from_docx(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Extract structured information
            resume_data = {
                'raw_text': text,
                'sections': self._extract_sections(text),
                'skills': self._extract_skills(text),
                'education': self._extract_education(text),
                'experience': self._extract_experience(text),
                'contact_info': self._extract_contact_info(text)
            }
            
            return resume_data
            
        except Exception as e:
            logger.error(f"Error parsing resume: {str(e)}")
            raise

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            raise

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract main sections from resume text."""
        sections = {}
        current_section = "header"
        current_content = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a section header
            if line.upper() in ['EDUCATION', 'EXPERIENCE', 'SKILLS', 'PROJECTS', 'CERTIFICATIONS']:
                if current_section and current_content:
                    sections[current_section.lower()] = '\n'.join(current_content)
                current_section = line.lower()
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections[current_section.lower()] = '\n'.join(current_content)
        
        return sections

    def _extract_skills(self, text: str) -> list:
        """Extract skills from resume text."""
       
        skills_section = text.lower()
        common_skills = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php',
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
            'django', 'flask', 'spring', 'sql', 'nosql', 'mongodb', 'postgresql',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'agile',
            'scrum', 'machine learning', 'ai', 'data science', 'big data',
            'devops', 'ci/cd', 'testing', 'security', 'networking'
        ]
        
        found_skills = []
        for skill in common_skills:
            if skill in skills_section:
                found_skills.append(skill)
        
        return found_skills

    def _extract_education(self, text: str) -> list:
        """Extract education information from resume text."""
        
        education = []
        education_section = text.lower()
        
        if 'education' in education_section:
            edu_text = education_section.split('education')[1].split('\n\n')[0]
            education.append(edu_text.strip())
        
        return education

    def _extract_experience(self, text: str) -> list:
        """Extract work experience from resume text."""
        
        experience = []
        experience_section = text.lower()
        
        if 'experience' in experience_section:
            exp_text = experience_section.split('experience')[1].split('\n\n')[0]
            experience.append(exp_text.strip())
        
        return experience

    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from resume text."""
        contact_info = {}
        
        # Extract email
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Extract phone
        phone_pattern = r'\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact_info['phone'] = phones[0]
        
        return contact_info 