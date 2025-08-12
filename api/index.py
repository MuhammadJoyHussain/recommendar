import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier

# Load environment variables
load_dotenv()
app = Flask(__name__)
CORS(app)

# Connect to MongoDB
mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
jobdb = client['beacon-nest']

# Load job data
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

# Load models and encoders
model_dir = os.path.join(os.path.dirname(__file__), '../models')
clf = CatBoostClassifier()
clf.load_model(os.path.join(model_dir, "job_match_model.cbm"))
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

    if isinstance(candidate_skills, list):
        candidate_skills = " ".join(candidate_skills)
    elif not isinstance(candidate_skills, str):
        candidate_skills = ""

    try:
        education_encoded = le_edu.transform([candidate_education])[0]
    except ValueError:
        education_encoded = 0

    try:
        role_encoded = le_role.transform([candidate_role])[0]
    except ValueError:
        role_encoded = 0

    candidate_vec = tfidf_candidate.transform([candidate_skills]).toarray()

    results = []
    for _, job in jobs_df.iterrows():
        job_skills = job.get('job_skills', [])
        if isinstance(job_skills, list):
            job_skills = " ".join(job_skills)
        elif not isinstance(job_skills, str):
            job_skills = ""

        job_vec = tfidf_job.transform([job_skills]).toarray()

        features = np.hstack([
            candidate_vec,
            job_vec,
            np.array([[candidate_experience, education_encoded, role_encoded]])
        ])

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

    results = sorted(results, key=lambda x: x['match_probability'], reverse=True)
    top_results = results[:5]

    return jsonify(top_results)

@app.route('/recommend-users', methods=['POST'])
def recommend_users():
    data = request.get_json()

    job_skills = data.get('job_skills', [])
    job_education = data.get('education_level', '')
    job_experience = data.get('experience_years', 0)
    job_role = data.get('job_role', '')

    if isinstance(job_skills, list):
        job_skills = " ".join(job_skills)
    elif not isinstance(job_skills, str):
        job_skills = ""

    try:
        education_encoded = le_edu.transform([job_education])[0]
    except ValueError:
        education_encoded = 0

    try:
        role_encoded = le_role.transform([job_role])[0]
    except ValueError:
        role_encoded = 0

    job_vec = tfidf_job.transform([job_skills]).toarray()

    user_cursor = jobdb.users.find( {"role": "user"}, {
        "_id": 1,
        "firstName": 1,
        "lastName": 1,
        "email": 1,
        "skills": 1,
        "education_level": 1,
        "experience_years": 1,
        "job_role": 1
    })
    users = list(user_cursor)
    for user in users:
        user['_id'] = str(user['_id'])

    results = []
    for user in users:
        user_skills = user.get('skills', [])
        if isinstance(user_skills, list):
            user_skills = " ".join(user_skills)
        elif not isinstance(user_skills, str):
            user_skills = ""

        user_vec = tfidf_candidate.transform([user_skills]).toarray()
        user_experience = user.get('experience_years', 0)
        user_edu = user.get('education_level', '')
        user_role = user.get('job_role', '')

        try:
            user_edu_encoded = le_edu.transform([user_edu])[0]
        except ValueError:
            user_edu_encoded = 0

        try:
            user_role_encoded = le_role.transform([user_role])[0]
        except ValueError:
            user_role_encoded = 0

        features = np.hstack([
            user_vec,
            job_vec,
            np.array([[user_experience, user_edu_encoded, user_role_encoded]])
        ])

        pred_proba = clf.predict_proba(features)[0][1]

        results.append({
            'user_id': user['_id'],
            'name': f"{user.get('firstName', '')} {user.get('lastName', '')}",
            'email': user.get('email', ''),
            'match_probability': float(pred_proba)
        })

    results = sorted(results, key=lambda x: x['match_probability'], reverse=True)
    top_users = results[:5]

    return jsonify(top_users)

