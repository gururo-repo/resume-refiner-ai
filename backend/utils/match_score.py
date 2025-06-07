import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .model_loader import match_model, feature_scaler, model_features, sentence_transformer
from .skill_matcher import extract_skills, SKILLS_DB
import re

def get_match_score(resume_text: str, job_description: str) -> dict:
    """
    Calculate match score between resume and job description.
    Returns a dictionary containing match percentage, ATS score, strengths, weaknesses, and recommendations.
    """
    # Extract skills from both texts with category information
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    
    # Calculate skill match metrics with category weights
    category_weights = {
        'programming': 1.0,
        'frameworks': 0.9,
        'cloud': 0.9,
        'data_science': 0.9,
        'devops': 0.8,
        'security': 0.8,
        'tools': 0.7,
        'soft_skills': 0.6
    }
    
    # Calculate weighted skill matches
    resume_skill_set = set()
    job_skill_set = set()
    weighted_resume_skills = {}
    weighted_job_skills = {}
    
    for category, skills in resume_skills.items():
        for skill in skills:
            resume_skill_set.add(skill)
            weighted_resume_skills[skill] = category_weights.get(category, 0.5)
    
    for category, skills in job_skills.items():
        for skill in skills:
            job_skill_set.add(skill)
            weighted_job_skills[skill] = category_weights.get(category, 0.5)
    
    # Calculate weighted Jaccard similarity
    if len(resume_skill_set) == 0 or len(job_skill_set) == 0:
        jaccard_similarity = 0.0
    else:
        intersection = resume_skill_set.intersection(job_skill_set)
        union = resume_skill_set.union(job_skill_set)
        
        # Calculate weighted intersection and union
        weighted_intersection = sum(weighted_job_skills[skill] for skill in intersection)
        weighted_union = sum(weighted_job_skills[skill] for skill in union)
        
        jaccard_similarity = weighted_intersection / weighted_union if weighted_union > 0 else 0.0
    
    # Calculate common skills with weights
    common_skills = resume_skill_set.intersection(job_skill_set)
    common_skills_count = len(common_skills)
    weighted_common_skills = sum(weighted_job_skills[skill] for skill in common_skills)
    
    # Get embeddings for semantic similarity with improved context
    resume_sections = {
        'experience': extract_section(resume_text, 'experience'),
        'skills': extract_section(resume_text, 'skills'),
        'projects': extract_section(resume_text, 'projects')
    }
    
    job_sections = {
        'requirements': extract_section(job_description, 'requirements'),
        'responsibilities': extract_section(job_description, 'responsibilities'),
        'qualifications': extract_section(job_description, 'qualifications')
    }
    
    # Calculate section-wise semantic similarity
    section_similarities = {}
    for resume_section, resume_content in resume_sections.items():
        if not resume_content:
            continue
        for job_section, job_content in job_sections.items():
            if not job_content:
                continue
            resume_embedding = sentence_transformer.encode([resume_content])[0]
            job_embedding = sentence_transformer.encode([job_content])[0]
            similarity = np.dot(resume_embedding, job_embedding) / (
                np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
            )
            section_similarities[f"{resume_section}_{job_section}"] = similarity
    
    # Calculate overall semantic similarity
    resume_embedding = sentence_transformer.encode([resume_text])[0]
    job_embedding = sentence_transformer.encode([job_description])[0]
    skill_vector_similarity = np.dot(resume_embedding, job_embedding) / (
        np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
    )
    
    # Calculate additional features
    skill_count_diff = abs(len(resume_skill_set) - len(job_skill_set))
    skill_count_ratio = weighted_common_skills / max(1, sum(weighted_job_skills.values()))
    skills_density = len(resume_skill_set) / max(1, len(resume_text.split()))
    
    # Prepare features for model prediction
    features = np.array([[
        jaccard_similarity,
        weighted_common_skills,
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
    
    # Calculate ATS score with improved weighting
    ats_score = min(100, max(0, (
        jaccard_similarity * 0.4 +
        skill_vector_similarity * 0.3 +
        sum(section_similarities.values()) / max(1, len(section_similarities)) * 0.3
    ) * 100))
    
    # Identify strengths and weaknesses with category information
    strengths = []
    for skill in common_skills:
        category = next((cat for cat, skills in job_skills.items() if skill in skills), 'other')
        strengths.append(f"Strong {category} skill: {skill}")
    
    weaknesses = []
    for skill in job_skill_set - resume_skill_set:
        category = next((cat for cat, skills in job_skills.items() if skill in skills), 'other')
        weaknesses.append(f"Missing {category} skill: {skill}")
    
    # Generate recommendations
    recommendations = []
    
    # Skill gaps
    if weaknesses:
        top_weaknesses = [w.split(': ')[1] for w in weaknesses[:5]]
        recommendations.append(f"Add missing skills: {', '.join(top_weaknesses)}")
    
    # Skills density
    if skills_density < 0.1:
        recommendations.append("Increase skills density in resume by adding more relevant skills")
    
    # Section improvements
    for section, similarity in section_similarities.items():
        if similarity < 0.5:
            recommendations.append(f"Improve alignment in {section.replace('_', ' ')} section")
    
    # Formatting suggestions
    if len(resume_text.split()) < 200:
        recommendations.append("Add more detailed content to resume")
    
    return {
        'match_score': round(match_percentage, 2),
        'ats_score': round(ats_score, 2),
        'strengths': strengths[:5],
        'weaknesses': weaknesses[:5],
        'recommendations': recommendations[:5]
    }

def extract_section(text: str, section_name: str) -> str:
    """Extract content from a specific section of the text."""
    section_patterns = [
        fr'\n\s*{section_name}.*?\n(.*?)(?=\n\s*[A-Z][A-Z\s]+:|$)',
        fr'\n\s*{section_name}.*?\n(.*?)(?=\n\s*\d+\.|$)',
        fr'\n\s*{section_name}.*?\n(.*?)(?=\n\s*[-•\*]|$)'
    ]
    
    for pattern in section_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return "" 