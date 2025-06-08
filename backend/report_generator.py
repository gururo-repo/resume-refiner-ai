import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_report(resume_text, analysis_results):
    """
    Generate a comprehensive report from resume analysis results.
    
    Args:
        resume_text (str): The parsed resume text
        analysis_results (dict): Dictionary containing analysis results including:
            - match_score
            - suggested_roles
            - skill_matches
            - ats_score
            - improvement_suggestions
    
    Returns:
        dict: A structured report with all analysis results
    """
    try:
        logger.info("Generating resume analysis report")
        
        # Create timestamp for the report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Structure the report
        report = {
            "timestamp": timestamp,
            "resume_summary": {
                "length": len(resume_text),
                "word_count": len(resume_text.split()),
                "line_count": len(resume_text.splitlines())
            },
            "analysis_results": {
                "match_score": analysis_results.get("match_score", 0),
                "suggested_roles": analysis_results.get("suggested_roles", []),
                "skill_matches": analysis_results.get("skill_matches", []),
                "ats_score": analysis_results.get("ats_score", 0),
                "improvement_suggestions": analysis_results.get("improvement_suggestions", [])
            },
            "metadata": {
                "version": "1.0",
                "analysis_type": "comprehensive"
            }
        }
        
        logger.info("Successfully generated report")
        return report
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise ValueError(f"Failed to generate report: {str(e)}")

def format_report_for_display(report):
    """
    Format the report for display in the frontend.
    
    Args:
        report (dict): The generated report
        
    Returns:
        dict: Formatted report with sections ready for display
    """
    try:
        formatted = {
            "summary": {
                "timestamp": report["timestamp"],
                "resume_stats": report["resume_summary"]
            },
            "analysis": {
                "match_score": {
                    "score": report["analysis_results"]["match_score"],
                    "label": "Job Match Score",
                    "description": "How well your resume matches the job requirements"
                },
                "ats_score": {
                    "score": report["analysis_results"]["ats_score"],
                    "label": "ATS Score",
                    "description": "How well your resume performs with Applicant Tracking Systems"
                },
                "suggested_roles": {
                    "roles": report["analysis_results"]["suggested_roles"],
                    "label": "Suggested Roles",
                    "description": "Job roles that match your skills and experience"
                },
                "skill_matches": {
                    "matches": report["analysis_results"]["skill_matches"],
                    "label": "Skill Matches",
                    "description": "Skills found in your resume that match job requirements"
                },
                "improvements": {
                    "suggestions": report["analysis_results"]["improvement_suggestions"],
                    "label": "Improvement Suggestions",
                    "description": "Recommendations to improve your resume"
                }
            }
        }
        
        return formatted
        
    except Exception as e:
        logger.error(f"Error formatting report: {str(e)}")
        raise ValueError(f"Failed to format report: {str(e)}") 