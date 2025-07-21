import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd
import numpy as np
import joblib

load_dotenv()
app = Flask(__name__)
CORS(app)

# Mongo connection
mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
jobdb = client['beacon-nest']

# Load job data from MongoDB
job_cursor = jobdb.vacancies.find({}, {
    "_id": 1,
    "title": 1,
    "companyOverview": 1,
    "jobSummary": 1,
    "type": 1,
    "company": 1,
    "location": 1,
    "job_skills": 1,
    "education_level": 1,
    "job_role": 1,
    "experience_years": 1
})
jobs = list(job_cursor)
for job in jobs:
    job['_id'] = str(job['_id'])

jobs_df = pd.DataFrame(jobs)

# Load ML models and encoders
model_dir = os.path.join(os.path.dirname(__file__), '../models')
clf = joblib.load(os.path.join(model_dir, "job_match_model.pkl"))
tfidf_candidate = joblib.load(os.path.join(model_dir, "vectorizer_candidate.pkl"))
tfidf_job = joblib.load(os.path.join(model_dir, "vectorizer_job.pkl"))
le_edu = joblib.load(os.path.join(model_dir, "le_education.pkl"))
le_role = joblib.load(os.path.join(model_dir, "le_role.pkl"))

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()

    candidate_skills = data.get('skills', [])
    candidate_education = data.get('education_level', '')
    candidate_experience = data.get('experience_years', 0)
    candidate_role = data.get('job_role', '')

    # Convert skills list to space-separated string for TF-IDF vectorizer
    if isinstance(candidate_skills, list):
        candidate_skills = " ".join(candidate_skills)
    elif not isinstance(candidate_skills, str):
        candidate_skills = ""

    # Encode candidate education and role, handle unseen labels gracefully
    try:
        education_encoded = le_edu.transform([candidate_education])[0]
    except ValueError:
        education_encoded = 0

    try:
        role_encoded = le_role.transform([candidate_role])[0]
    except ValueError:
        role_encoded = 0

    # Vectorize candidate skills
    candidate_vec = tfidf_candidate.transform([candidate_skills]).toarray()

    results = []
    for idx, job in jobs_df.iterrows():
        job_skills = job.get('job_skills', [])
        if isinstance(job_skills, list):
            job_skills = " ".join(job_skills)
        elif not isinstance(job_skills, str):
            job_skills = ""

        job_vec = tfidf_job.transform([job_skills]).toarray()

        # Combine features exactly like training
        features = np.hstack([
            candidate_vec,
            job_vec,
            np.array([[candidate_experience, education_encoded, role_encoded]])
        ])

        # Predict probability of candidate being selected for this job
        pred_proba = clf.predict_proba(features)[0][1]

        results.append({
            'job_id': job['_id'],
            'title': job.get('title', ''),
            'company': job.get('company', ''),
            'type': job.get('type', ''),
            'location': job.get('location', ''),
            'summary': job.get('jobSummary', ''),
            'match_probability': float(pred_proba)
        })

    # Sort jobs by match probability descending and return top 5
    results = sorted(results, key=lambda x: x['match_probability'], reverse=True)
    top_results = results[:5]

    return jsonify(top_results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
