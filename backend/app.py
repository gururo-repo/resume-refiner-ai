from flask import Flask, request, jsonify
from utils.match_score import get_match_score
from utils.role_predictor import predict_roles
from utils.skill_matcher import extract_and_match_skills
from utils.genai_suggester import suggest_resume
from utils.resume_parser import parse_resume_file

app = Flask(__name__)

@app.route('/upload-resume', methods=['POST'])
def upload_resume():
    file = request.files.get('resume')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    resume_text = parse_resume_file(file)
    return jsonify({'resume_text': resume_text})

@app.route('/upload-jd', methods=['POST'])
def upload_jd():
    jd_text = request.json.get('jd_text')
    if not jd_text:
        return jsonify({'error': 'No JD text provided'}), 400
    return jsonify({'jd_text': jd_text})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    resume_text = data.get('resume_text')
    jd_text = data.get('jd_text')
    if not resume_text or not jd_text:
        return jsonify({'error': 'Missing resume or JD'}), 400
    match_score = get_match_score(resume_text, jd_text)
    missing_skills = extract_and_match_skills(resume_text, jd_text)
    recommended_roles = predict_roles(resume_text)
    return jsonify({
        'match_pct': match_score['match_pct'],
        'ats_score': match_score['ats_score'],
        'missing_skills': missing_skills,
        'recommended_roles': recommended_roles,
        'strengths': match_score['strengths'],
        'weaknesses': match_score['weaknesses'],
        'areas_to_improve': match_score['areas_to_improve']
    })

@app.route('/genai-suggest', methods=['POST'])
def genai_suggest():
    data = request.json
    resume_text = data.get('resume_text')
    jd_text = data.get('jd_text')
    if not resume_text or not jd_text:
        return jsonify({'error': 'Missing resume or JD'}), 400
    suggestion = suggest_resume(resume_text, jd_text)
    return jsonify({'suggested_resume': suggestion})

if __name__ == '__main__':
    app.run(debug=True) 