import joblib
import os
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models')
role_clf = joblib.load(os.path.join(MODEL_DIR, 'role_classifier.pkl'))
encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))

def predict_roles(resume_text):
    X = tfidf.transform([resume_text])
    probs = role_clf.predict_proba(X)[0]
    top_idx = probs.argsort()[-3:][::-1]
    return [encoder.classes_[i] for i in top_idx] 