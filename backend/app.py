from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
import logging
import traceback
from utils.resume_parser import ResumeParser
from utils.model_loader import get_model_loader
from utils.match_score import MatchScoreCalculator
from utils.role_predictor import RolePredictor
from dotenv import load_dotenv
import re

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

# Enhanced CORS Configuration - FIXED
CORS(app, 
     origins=['https://resume-refiner-ai.vercel.app', 'http://localhost:3000', 'http://localhost:5173'],
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'OPTIONS'],
     supports_credentials=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_job_description(job_description):
    """
    Validates if the job description is meaningful and appropriate.
    Returns (is_valid, message) tuple.
    """
    if not job_description:
        return False, "Job description is required"
    
    if len(job_description.strip()) < 50:
        return False, "Job description too short"
    
    # Check for minimum required sections
    required_sections = ['responsibilities', 'requirements', 'qualifications']
    found_sections = [section for section in required_sections 
                     if re.search(rf'\b{section}\b', job_description.lower())]
    
    if len(found_sections) < 2:
        return False, "Missing required sections"
    
    # Check for minimum content length
    if len(job_description.split()) < 100:
        return False, "Job description needs more details"
    
    # Check for inappropriate content
    inappropriate_patterns = [
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        r'[<>]',
        r'[^\w\s.,;:!?()\-@#$%&*+/=]'
    ]
    
    for pattern in inappropriate_patterns:
        if re.search(pattern, job_description):
            return False, "Invalid content detected"
    
    return True, "Job description is valid"

# Add a root route to handle the HEAD request
@app.route('/')
def root():
    return jsonify({'message': 'Resume Refiner API is running', 'version': '1.0'})

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    
    try:
        logger.info("Received analyze request")
        logger.info(f"Request files: {list(request.files.keys())}")
        logger.info(f"Request form: {list(request.form.keys())}")
        logger.info(f"Request origin: {request.headers.get('Origin', 'No origin header')}")
        
        # Check if file is present
        if 'resume' not in request.files:
            logger.error("No resume file in request")
            return jsonify({'error': 'No resume file provided'}), 400
        
        file = request.files['resume']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload PDF, DOC, or DOCX files only.'}), 400
        
        # Get and validate job description
        job_description = request.form.get('job_description', '')
        is_valid, message = validate_job_description(job_description)
        if not is_valid:
            logger.error(f"Invalid job description: {message}")
            return jsonify({'error': message}), 400
        
        logger.info(f"Processing file: {file.filename}")
        logger.info(f"Job description length: {len(job_description)}")
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(filepath)
        logger.info(f"File saved to: {filepath}")
        
        try:
            # Parse resume
            parser = ResumeParser()
            resume_data = parser.parse_resume(filepath)
            logger.info("Resume parsed successfully")
            
            # Validate resume format
            is_valid, message = parser.validate_resume(resume_data.get('raw_text', ''))
            if not is_valid:
                logger.error(f"Resume validation failed: {message}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': message}), 400
            
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
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info("Temporary file cleaned up")
            
            logger.info("Successfully completed resume analysis")
            return jsonify(analysis_result)
            
        except Exception as e:
            logger.error(f"Error analyzing resume: {str(e)}")
            logger.error(traceback.format_exc())
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error in analyze_resume endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'version': '1.0',
        'port': os.environ.get('PORT', 10000)
    })

# Add error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

# Add after_request handler for additional CORS headers if needed
@app.after_request
def after_request(response):
    # Log the response headers for debugging
    logger.info(f"Response headers: {dict(response.headers)}")
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False, timeout=120)  # Increased timeout to 120 seconds