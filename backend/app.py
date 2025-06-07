from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import traceback
from utils.resume_parser import parse_resume_file
from utils.genai_suggester import ResumeImprover
from utils.match_score import get_match_score
from utils.role_predictor import predict_roles
from utils.skill_matcher import extract_and_match_skills
from report_generator import generate_report

app = Flask(__name__)
CORS(app)

# Initialize ResumeImprover
resume_improver = ResumeImprover()

@app.route('/api/analyze', methods=['POST'])
def analyze_resume_endpoint():
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        resume_file = request.files['resume']
        job_description = request.form.get('job_description', '')
        
        if not job_description:
            return jsonify({'error': 'No job description provided'}), 400
        
        # Parse resume file
        resume_text = parse_resume_file(resume_file)
        
        # Get match score
        match_analysis = get_match_score(resume_text, job_description)
        
        # Get role predictions
        roles = predict_roles(resume_text)
        
        # Get skill matches
        skill_analysis = extract_and_match_skills(resume_text, job_description)
        
        # Get ATS analysis and improvements
        ats_analysis = resume_improver.analyze_ats_compatibility(resume_text, job_description)
        improvements = resume_improver.improve_resume(resume_text, job_description)
        
        # Combine all analyses
        analysis = {
            'match_score': match_analysis['match_score'],
            'ats_score': ats_analysis['analysis']['ats_score'],
            'strengths': match_analysis['strengths'],
            'weaknesses': match_analysis['weaknesses'],
            'recommendations': match_analysis['recommendations'],
            'predicted_roles': roles,
            'skill_matches': skill_analysis['matched_skills'],
            'missing_skills': skill_analysis['missing_skills'],
            'improved_resume': improvements['improved_resume']
        }
        
        return jsonify(analysis)
    
    except Exception as e:
        print(f"Error in analyze_resume_endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report_endpoint():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['match_score', 'ats_score', 'strengths', 'weaknesses', 'recommendations']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Generate PDF report
        pdf_path = generate_report(data)
        
        # Send the PDF file
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='resume_analysis_report.pdf'
        )
    
    except Exception as e:
        print(f"Error in generate_report_endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True) 