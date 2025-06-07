import re
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sparse
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher

def _check_section_headers(resume_text, section_keywords):
    """Helper function to check for section headers"""
    for keyword in section_keywords:
        section_header_patterns = [
            fr'\b({keyword})\s*:',  
            fr'\n\s*({keyword})\s*\n',  
            fr'\n\s*({keyword})\s*[^\n]*\n\s*[-•\*]'  
        ]

        for pattern in section_header_patterns:
            if re.search(pattern, resume_text, re.IGNORECASE):
                return True
    return False

def check_sections(resume_text):
    """Calculate score based on presence of essential resume sections with improved detection"""
    sections = {
        'experience': [
            'experience', 'work history', 'professional background', 'employment', 
            'work experience', 'career history', 'professional experience',
            'employment history', 'work background', 'professional journey'
        ],
        'education': [
            'education', 'academic', 'qualification', 'degree', 'university', 
            'college', 'school', 'certification', 'academic background',
            'educational background', 'academic qualifications', 'degrees',
            'certifications', 'training', 'courses'
        ],
        'skills': [
            'skills', 'abilities', 'competencies', 'expertise', 'proficiencies', 
            'technical skills', 'core competencies', 'technical expertise',
            'professional skills', 'key skills', 'skill set', 'capabilities',
            'technical proficiencies', 'areas of expertise'
        ],
        'summary': [
            'summary', 'profile', 'objective', 'about me', 'professional summary', 
            'career objective', 'professional profile', 'executive summary',
            'career summary', 'personal statement', 'professional overview',
            'career profile', 'professional statement'
        ],
        'projects': [
            'projects', 'portfolio', 'project experience', 'project history',
            'project work', 'project portfolio', 'project showcase',
            'project achievements', 'project highlights', 'project details'
        ],
        'achievements': [
            'achievements', 'accomplishments', 'awards', 'recognition',
            'honors', 'certifications', 'professional achievements',
            'key achievements', 'notable accomplishments', 'awards and recognition'
        ]
    }
    
    section_scores = {}
    for section_name, section_keywords in sections.items():
        # Check for section headers with improved patterns
        header_patterns = [
            fr'\b({keyword})\s*:',  
            fr'\n\s*({keyword})\s*\n',  
            fr'\n\s*({keyword})\s*[^\n]*\n\s*[-•\*]',
            fr'\n\s*({keyword})\s*[^\n]*\n\s*[A-Z]',
            fr'\n\s*({keyword})\s*[^\n]*\n\s*\d+\.'
        ]
        
        header_detected = False
        for keyword in section_keywords:
            for pattern in header_patterns:
                if re.search(pattern.format(keyword=keyword), resume_text, re.IGNORECASE):
                    header_detected = True
                    break
            if header_detected:
                break
        
        # Check for content with improved detection
        content_detected = False
        for keyword in section_keywords:
            if keyword in resume_text.lower():
                # Check if keyword is part of a meaningful section
                context = re.search(fr'\n.*?{keyword}.*?\n', resume_text, re.IGNORECASE)
                if context and len(context.group().strip()) > len(keyword) + 5:
                    content_detected = True
                    break
        
        # Calculate section score
        if header_detected:
            section_scores[section_name] = 1.0
        elif content_detected:
            section_scores[section_name] = 0.7
        else:
            section_scores[section_name] = 0.0
    
    # Calculate weighted total score
    weights = {
        'experience': 0.25,
        'education': 0.20,
        'skills': 0.20,
        'summary': 0.15,
        'projects': 0.10,
        'achievements': 0.10
    }
    
    total_score = sum(section_scores.get(section, 0) * weight 
                     for section, weight in weights.items())
    
    return total_score

def check_keywords(resume_text, job_keywords):
    """Calculate score based on keyword matches with improved relevance detection"""
    if not job_keywords:
        return 0
    
    # Preprocess resume text for better matching
    clean_resume = re.sub(r'[.,;:!?()\[\]{}]', ' ', resume_text)
    clean_resume = re.sub(r'\s+', ' ', clean_resume).strip()
    
    # Enhanced keyword importance weights
    keyword_importance = {}
    for kw in job_keywords:
        base_weight = 1.0
        
        # Technical skills get higher weight
        if any(tech in kw.lower() for tech in [
            'python', 'java', 'javascript', 'react', 'aws', 'cloud', 'ml', 'ai',
            'docker', 'kubernetes', 'sql', 'nosql', 'devops', 'security'
        ]):
            base_weight += 0.5
        
        # Framework and library skills
        if any(fw in kw.lower() for fw in [
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express',
            'tensorflow', 'pytorch', 'scikit-learn'
        ]):
            base_weight += 0.3
        
        # Cloud and infrastructure skills
        if any(cloud in kw.lower() for cloud in [
            'aws', 'azure', 'gcp', 'cloud', 's3', 'ec2', 'lambda', 'kubernetes',
            'docker', 'terraform'
        ]):
            base_weight += 0.4
        
        keyword_importance[kw.lower()] = base_weight
    
    total_weight = sum(keyword_importance.values())
    
    # Enhanced keyword matching
    matches = {}
    for kw, weight in keyword_importance.items():
        # Exact match
        if f" {kw} " in f" {clean_resume} ":
            matches[kw] = weight
            continue
        
        # Contextual match for multi-word keywords
        if ' ' in kw:
            kw_parts = kw.split()
            
            # All parts present in close proximity
            if all(part in clean_resume for part in kw_parts if len(part) > 3):
                # Check if parts are within 5 words of each other
                parts_positions = [clean_resume.find(part) for part in kw_parts if len(part) > 3]
                if all(pos != -1 for pos in parts_positions):
                    max_distance = max(parts_positions) - min(parts_positions)
                    if max_distance < 50:  # Words are close to each other
                        matches[kw] = weight * 0.9
                        continue
            
            # Most parts present
            if len(kw_parts) > 2 and sum(1 for part in kw_parts if part in clean_resume and len(part) > 3) >= len(kw_parts) * 0.7:
                matches[kw] = weight * 0.7
                continue
        
        # Fuzzy match for single words
        if len(kw.split()) == 1 and len(kw) > 3:
            words = clean_resume.split()
            for word in words:
                if len(word) > 3 and get_skill_similarity(kw, word) > 0.8:
                    matches[kw] = weight * 0.6
                    break
    
    # Calculate weighted score with density bonus
    if total_weight == 0:
        return 0
    
    weighted_score = sum(matches.values()) / total_weight
    
    # Enhanced density bonus
    keyword_density = len(matches) / max(1, len(job_keywords))
    density_bonus = min(0.3, keyword_density * 0.5)  # Up to 30% bonus
    
    # Context bonus
    context_bonus = 0.0
    if len(matches) > 0:
        # Check if matched keywords appear in relevant sections
        relevant_sections = ['experience', 'skills', 'projects']
        for section in relevant_sections:
            section_match = re.search(fr'\n\s*{section}.*?\n', resume_text, re.IGNORECASE)
            if section_match:
                section_text = resume_text[section_match.start():section_match.end()].lower()
                section_matches = sum(1 for kw in matches if kw in section_text)
                if section_matches > 0:
                    context_bonus += 0.1  # 10% bonus per relevant section
    
    return min(1.0, weighted_score + density_bonus + context_bonus)

def get_skill_similarity(skill1: str, skill2: str) -> float:
    """Calculate similarity between two skills using SequenceMatcher."""
    return SequenceMatcher(None, skill1.lower(), skill2.lower()).ratio()

def check_formatting(resume_text):
    """Calculate score based on resume formatting with improved analysis"""
    format_score = 0
    
    # 1. Bullet points - comprehensive detection
    bullet_count = resume_text.count("•") + resume_text.count("- ") + resume_text.count("* ")
    numbered_bullets = len(re.findall(r'\n\s*\d+\.', resume_text))
    total_bullets = bullet_count + numbered_bullets
    
    # Scale bullet points score
    if total_bullets >= 10:
        format_score += 0.35  
    elif total_bullets >= 5:
        format_score += 0.25
    elif total_bullets > 0:
        format_score += 0.15
    
    # 2. Section headers detection 
    header_patterns = [
        r'(\n[A-Z][A-Z\s]+:)',  
        r'(\n[A-Z][A-Z\s]+\n)',  
        r'(\n[A-Z][a-z]+\s[A-Z][a-z]+:)',
        r'(\n\s*\d+\.\s*[A-Z][a-z]+)'  
    ]
    
    headers_count = 0
    for pattern in header_patterns:
        headers_count += len(re.findall(pattern, resume_text))
    
    # Scale headers score
    if headers_count >= 5:
        format_score += 0.35
    elif headers_count >= 3:
        format_score += 0.25
    elif headers_count > 0:
        format_score += 0.15
    
    # 3. Consistent date formatting
    date_patterns = [
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  
        r'\b\d{2}/\d{2}/\d{4}\b',  
        r'\b\d{4}-\d{2}-\d{2}\b',  #
        r'\b\d{4}\s*-\s*(?:Present|Current|Now)\b' 
    ]
    
    date_formats_found = 0
    for pattern in date_patterns:
        if re.search(pattern, resume_text):
            date_formats_found += 1

    date_format_score = min(0.15, date_formats_found * 0.05)
    format_score += date_format_score
    
    # 4. Contact information and links
    contact_patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b(?:\+\d{1,3}\s?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Phone
        r'linkedin\.com/in/[a-zA-Z0-9_-]+',  # LinkedIn
        r'github\.com/[a-zA-Z0-9_-]+'  # GitHub
    ]
    
    contact_score = 0
    for pattern in contact_patterns:
        if re.search(pattern, resume_text):
            contact_score += 0.05
    
    format_score += min(0.15, contact_score)
    
    return min(format_score, 1.0)

def check_context_relevance(resume_text, job_keywords):
    """Analyze the contextual relevance of keywords"""
    if not job_keywords:
        return 0

    context_score = 0
    key_sections = ['experience', 'project', 'skill', 'education']

    for section in key_sections:
       
        section_match = re.search(fr'\n\s*{section}.*?\n', resume_text, re.IGNORECASE)
        if not section_match:
            continue
            
        section_start = section_match.start()
        next_section_match = re.search(r'\n\s*[A-Z][A-Z\s]+\s*(?::|$)', resume_text[section_start+1:], re.IGNORECASE)
        
        if next_section_match:
            section_end = section_start + 1 + next_section_match.start()
        else:
            section_end = len(resume_text)
            
        section_text = resume_text[section_start:section_end].lower()
       
        section_keywords = sum(1 for kw in job_keywords if kw.lower() in section_text)
        
        # Add to context score based on section relevance
        weight = 0.4 if section in ['experience', 'skill'] else 0.2
        context_score += (section_keywords / len(job_keywords)) * weight
    
    return min(context_score, 1.0)

def ats_score(resume_text, job_keywords):
    """Calculate ATS compatibility score for a resume with enhanced algorithms"""
    if not resume_text or not job_keywords:
        return 0.0
    
    resume_text = resume_text.lower()
    job_keywords = [kw.strip() for kw in job_keywords if kw.strip()]
    
    # Calculate component scores
    section_score = check_sections(resume_text)
    keyword_score = check_keywords(resume_text, job_keywords)
    format_score = check_formatting(resume_text)
    context_score = check_context_relevance(resume_text, job_keywords)
    
    # Calculate final score with optimized weights
    final_score = (
        section_score * 0.25 + 
        keyword_score * 0.35 + 
        format_score * 0.20 + 
        context_score * 0.20
    )
    
    return round(final_score * 100, 2)
