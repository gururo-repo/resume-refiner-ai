from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
import logging
import traceback
from utils.resume_parser import ResumeParser
from utils.job_analyzer import analyze_job_description
from utils.model_loader import get_model_loader
from utils.skill_matcher import SkillMatcher
from utils.match_score import MatchScoreCalculator
from utils.role_predictor import RolePredictor
from utils.ats_scorer import calculate_ats_score
from utils.job_matcher import calculate_job_match
from utils.role_recommender import get_role_recommendations
from utils.skill_analyzer import analyze_missing_skills
from utils.genai_feedback import get_genai_feedback
from utils.ml_fallback import get_ml_fallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    try:
        # Check if files are present
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        resume_file = request.files['resume']
        if resume_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(resume_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get job description
        job_description = request.form.get('job_description', '')
        if not job_description:
            return jsonify({'error': 'No job description provided'}), 400
        
        # Save resume file
        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(filepath)
        
        # Parse resume
        parser = ResumeParser()
        resume_data = parser.parse_resume(filepath)
        
        # Analyze job description
        job_analysis = analyze_job_description(job_description)
        
        # Calculate match score
        match_calculator = MatchScoreCalculator()
        match_score = match_calculator.calculate_match_score(resume_data, job_description)
        match_components = match_calculator.analyze_match_components(resume_data, job_description)
        
        # Predict role
        role_predictor = RolePredictor()
        role_category, role_confidence = role_predictor.predict_role(job_description)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'resume_data': resume_data,
            'job_analysis': job_analysis,
            'match_score': match_score,
            'match_components': match_components,
            'role_prediction': {
                'category': role_category,
                'confidence': role_confidence
            }
        })
        
    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

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
        logger.error(f"Error in generate_report_endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 