from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
jobdb = client["beacon-nest"]

# Load and preprocess job data
job_cursor = jobdb.vacancies.find({}, {
    "_id": 1, "title": 1, "companyOverview": 1, "jobSummary": 1,
    "type": 1, "company": 1, "location": 1
})
jobs = list(job_cursor)
for job in jobs:
    job["_id"] = str(job["_id"])

def combine_text(row):
    return " ".join([
        str(row.get('title', '')), str(row.get('companyOverview', '')),
        str(row.get('jobSummary', '')), str(row.get('type', '')),
        str(row.get('company', '')), str(row.get('location', ''))
    ])

jobs_df = pd.DataFrame(jobs)
jobs_df['combined_text'] = jobs_df.apply(combine_text, axis=1)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(jobs_df['combined_text'])

# Endpoint: /recommend
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    skills = data.get("skills")

    if not skills:
        return jsonify({"error": "Skills input is required"}), 400

    user_input = " ".join(skills) if isinstance(skills, list) else skills
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    job_indices = similarity[0].argsort()[::-1]
    results = jobs_df.iloc[job_indices].head(3).to_dict(orient='records')

    return jsonify(results)

# Export app for Vercel
handler = app
