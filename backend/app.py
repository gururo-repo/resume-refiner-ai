from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.match_score import get_match_score
from utils.role_predictor import predict_roles
from utils.skill_matcher import extract_and_match_skills
from utils.genai_suggester import analyze_resume
from utils.resume_parser import parse_resume_file
from utils.model_loader import match_model, feature_scaler, model_features, sentence_transformer
import base64
from reportlab.pdfgen import canvas
from io import BytesIO

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze-resume', methods=['POST'])
def analyze_resume_endpoint():
    try:
        # Handle file upload
        if 'resume_file' in request.files:
            file = request.files['resume_file']
            resume_text = parse_resume_file(file)
        else:
            resume_text = request.form.get('resume_text')
        
        job_description = request.form.get('job_description')
        
        if not resume_text or not job_description:
            return jsonify({'error': 'Missing resume or job description'}), 400
        
        # Get match score and analysis
        match_analysis = get_match_score(resume_text, job_description)
        
        # Get role predictions
        roles = predict_roles(resume_text)
        
        # Get skill analysis
        skill_analysis = extract_and_match_skills(resume_text, job_description)
        
        # Get AI suggestions
        ai_suggestions = analyze_resume(resume_text, job_description)
        
        # Combine all results
        results = {
            'ats_score': match_analysis.get('ats_score', 0),
            'match_score': match_analysis.get('match_score', 0),
            'role_match': roles,
            'skill_analysis': skill_analysis,
            'strengths': match_analysis.get('strengths', []),
            'weaknesses': match_analysis.get('weaknesses', []),
            'recommendations': match_analysis.get('recommendations', [])
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        
        # Create PDF
        buffer = BytesIO()
        p = canvas.Canvas(buffer)
        
        # Add content to PDF
        p.setFont("Helvetica-Bold", 16)
        p.drawString(50, 800, "Resume Analysis Report")
        
        p.setFont("Helvetica", 12)
        p.drawString(50, 750, f"ATS Score: {data.get('ats_score', 0)}%")
        p.drawString(50, 730, f"Match Score: {data.get('match_score', 0)}%")
        
        # Add strengths
        p.drawString(50, 700, "Strengths:")
        y = 680
        for strength in data.get('strengths', []):
            p.drawString(70, y, f"• {strength}")
            y -= 20
        
        # Add weaknesses
        p.drawString(50, y - 20, "Areas for Improvement:")
        y -= 40
        for weakness in data.get('weaknesses', []):
            p.drawString(70, y, f"• {weakness}")
            y -= 20
        
        # Add recommendations
        p.drawString(50, y - 20, "Recommendations:")
        y -= 40
        for rec in data.get('recommendations', []):
            p.drawString(70, y, f"• {rec}")
            y -= 20
        
        p.save()
        
        # Get PDF content
        pdf_content = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'pdf_content': pdf_content,
            'filename': 'resume_analysis_report.pdf'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 