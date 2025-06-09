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
from utils.groq_analyzer import get_groq_analysis
from utils.ml_fallback import get_ml_fallback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Check required environment variables
required_env_vars = ['GROQ_API_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
    logger.warning("Some features may be disabled")

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
        # Check if file is present
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        file = request.files['resume']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get job description
        job_description = request.form.get('job_description', '')
        if not job_description:
            return jsonify({'error': 'No job description provided'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(filepath)
        
        try:
            # Parse resume
            parser = ResumeParser()
            resume_data = parser.parse_resume(filepath)
            
            # Initialize model loader
            model_loader = get_model_loader()
            
            # Try Groq analysis first
            logger.info("Attempting to get analysis from Groq API...")
            groq_analysis = model_loader.try_groq_analysis(resume_data.get('raw_text', ''), job_description)
            
            if groq_analysis:
                logger.info("Successfully got analysis from Groq API")
                analysis_result = {
                    'ats_score': groq_analysis.get('ats_score', 0),
                    'job_match_score': groq_analysis.get('job_match_score', 0),
                    'strengths': groq_analysis.get('strengths', []),
                    'weaknesses': groq_analysis.get('weaknesses', []),
                    'improvements': groq_analysis.get('improvement_tips', []),
                    'format_analysis': groq_analysis.get('format_analysis', {}),
                    'skills_analysis': groq_analysis.get('skills_analysis', {}),
                    'role_prediction': {
                        'category': groq_analysis.get('role_match', {}).get('primary_role', 'Unknown'),
                        'confidence': groq_analysis.get('role_match', {}).get('match_confidence', 0) / 100.0
                    },
                    'analysis_source': 'groq'
                }
            else:
                logger.warning("Groq analysis failed, falling back to ML models")
                # Fallback to ML models
                match_calculator = MatchScoreCalculator()
                match_score = match_calculator.calculate_match_score(resume_data, job_description)
                match_components = match_calculator.analyze_match_components(resume_data, job_description)
                
                role_predictor = RolePredictor()
                role_prediction = role_predictor.predict_role(resume_data.get('raw_text', ''), job_description)
                
                analysis_result = {
                    'ats_score': match_components.get('ats_score', 0),
                    'job_match_score': match_score,
                    'strengths': match_components.get('strengths', []),
                    'weaknesses': match_components.get('weaknesses', []),
                    'improvements': match_components.get('improvements', []),
                    'format_analysis': match_components.get('format_analysis', {}),
                    'skills_analysis': match_components.get('skills_analysis', {}),
                    'role_prediction': role_prediction,
                    'analysis_source': 'local_models'
                }
            
            # Clean up
            os.remove(filepath)
            
            logger.info("Successfully completed resume analysis")
            return jsonify(analysis_result)
            
        except Exception as e:
            logger.error(f"Error analyzing resume: {str(e)}")
            logger.error(traceback.format_exc())
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        logger.error(f"Error in analyze_resume endpoint: {str(e)}")
        logger.error(traceback.format_exc())
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
    app.run(host='0.0.0.0', port=5000, debug=True)  # Enable debug mode for better error messages 