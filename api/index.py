import os, re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB") or 'beacon-nest'
JOBS_COLLECTION = os.getenv("JOBS_COLLECTION")  or 'vacancies'
CANDS_COLLECTION = os.getenv("CANDS_COLLECTION") or 'users'

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "xgb_recruitment_model.joblib"))
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model

def _to_text_list(x):
    if not x: return []
    if isinstance(x, list): return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str): return [i.strip() for i in re.split(r"[|,;/]+", x) if i.strip()]
    return []

def _years_from_experience(exp_items):
    if not exp_items: return 0.0
    years, now = 0.0, datetime.utcnow().date()
    for item in exp_items:
        sd = str(item.get("startDate") or "").split("T")[0]
        ed = str(item.get("endDate") or "").split("T")[0]
        try: sdate = datetime.fromisoformat(sd).date()
        except Exception: continue
        edate = now
        if ed:
            try: edate = datetime.fromisoformat(ed).date()
            except Exception: pass
        if edate > sdate:
            years += (edate - sdate).days / 365.25
    return round(max(years, 0.0), 2)

def _locations_match(cand_loc, job_loc):
    if not cand_loc or not job_loc: return 0
    c0, j0 = str(cand_loc).lower(), str(job_loc).lower()
    return int(any(part.strip() and part.strip() in j0 for part in re.split(r"[,/|-]", c0)))

def _parse_salary_range(s):
    if not s: return None, None, None
    nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", s)]
    cur = None
    m = re.search(r"\b(usd|gbp|eur|cad|aud|inr)\b", s.lower())
    if m: cur = m.group(1).upper()
    if len(nums) == 1: return nums[0], nums[0], cur
    if len(nums) >= 2: return min(nums[0], nums[1]), max(nums[0], nums[1]), cur
    return None, None, cur

def _salary_match(cand_expected, job_salary):
    if cand_expected is None or job_salary is None: return 0
    js_min, js_max, _ = _parse_salary_range(job_salary)
    if js_min is None: return 0
    try: ce = int(str(cand_expected).replace(",", ""))
    except Exception: return 0
    return int(js_min <= ce <= (js_max or js_min))

def _skill_overlap_sets(a, b):
    sa = set(_to_text_list(a))
    sb = set(_to_text_list(b))
    inter, union = sa & sb, sa | sb
    cnt = len(inter)
    jacc = round((cnt / len(union)) if union else 0.0, 4)
    return cnt, jacc, sorted(list(inter))

def _prepare_features(candidate, job):
    cand_skills = _to_text_list(candidate.get("skills"))
    job_skills = _to_text_list(job.get("skills")) \
               + _to_text_list(job.get("requiredQualifications")) \
               + _to_text_list(job.get("preferredQualifications"))
    cand_skills_str = "|".join(cand_skills)
    job_skills_str = "|".join(job_skills)

    exp_years = candidate.get("experience_years")
    if exp_years is None:
        exp_years = _years_from_experience(candidate.get("experience"))

    edu = candidate.get("education_level") or candidate.get("education") or "Unknown"
    role = job.get("title") or "Unknown"
    industry = job.get("industry") or job.get("department") or "Unknown"
    cand_loc = candidate.get("city") or candidate.get("country") or "Unknown"
    job_loc = job.get("location") or "Unknown"
    job_type = job.get("type") or "Unknown"
    dept = job.get("department") or "Unknown"

    loc_match = candidate.get("location_match")
    if loc_match is None:
        loc_match = _locations_match(cand_loc, job_loc)
    salary_match = _salary_match(candidate.get("expectedSalary"), job.get("salary"))

    overlap_count, jaccard, matched = _skill_overlap_sets(cand_skills_str, job_skills_str)

    features = pd.DataFrame([{
        "candidate_skills": cand_skills_str,
        "job_skills": job_skills_str,
        "experience_years": float(exp_years or 0.0),
        "education_level": str(edu),
        "job_role": str(role),
        "industry": str(industry),
        "candidate_location": str(cand_loc),
        "job_location": str(job_loc),
        "job_type": str(job_type),
        "department": str(dept),
        "location_match": int(loc_match),
        "salary_match": int(salary_match),
        "skill_overlap_count": int(overlap_count),
        "skill_jaccard": float(jaccard),
    }])

    return features, matched

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug/status")
def debug_status():
    try:
        return {
            "db": MONGO_DB,
            "jobs_collection": JOBS_COLLECTION,
            "cands_collection": CANDS_COLLECTION,
            "jobs_count": db[JOBS_COLLECTION].count_documents({}),
            "cands_count": db[CANDS_COLLECTION].count_documents({}),
            "has_model": os.path.exists(MODEL_PATH)
        }
    except Exception as e:
        return {"error": str(e)}, 500

@app.post("/recommend")
def recommend_jobs_for_user_payload():
    try:
        data = request.get_json(force=True) or {}
        candidate = {
            "skills": data.get("skills") or [],
            "experience": data.get("experience"),
            "experience_years": data.get("experience_years"),
            "education_level": data.get("education_level"),
            "city": data.get("city"),
            "country": data.get("country"),
            "expectedSalary": data.get("expectedSalary"),
            "location_match": data.get("location_match"),
        }

        jobs = list(db[JOBS_COLLECTION].find({}))
        if not jobs:
            body_jobs = data.get("jobs") or []
            jobs = [{
                "_id": j.get("_id") or j.get("id"),
                "title": j.get("title"),
                "company": j.get("company"),
                "location": j.get("location"),
                "type": j.get("type"),
                "department": j.get("department"),
                "salary": j.get("salary"),
                "industry": j.get("industry"),
                "skills": j.get("skills") or j.get("requiredQualifications") or [],
                "requiredQualifications": j.get("requiredQualifications") or [],
                "preferredQualifications": j.get("preferredQualifications") or [],
            } for j in body_jobs]

        mdl = get_model()
        scored = []
        for job in jobs:
            feats, matched = _prepare_features(candidate, job)
            prob = float(mdl.predict_proba(feats)[:, 1][0])
            scored.append({
                "job_id": str(job.get("_id")),
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "type": job.get("type"),
                "probability": prob,
                "matched_skills": matched,
            })
        scored.sort(key=lambda x: x["probability"], reverse=True)
        top_n = int(data.get("top_n", 5))
        return jsonify({"recommendations": scored[:top_n]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/recommend-users")
def recommend_users_for_job_payload():
    try:
        data = request.get_json(force=True) or {}
        job = {
            "skills": data.get("skills") or [],
            "requiredQualifications": data.get("requiredQualifications") or [],
            "preferredQualifications": data.get("preferredQualifications") or [],
            "title": data.get("title"),
            "location": data.get("location"),
            "type": data.get("type"),
            "department": data.get("department"),
            "salary": data.get("salary"),
            "industry": data.get("industry"),
        }

        candidates = list(db[CANDS_COLLECTION].find({}))
        if not candidates:
            body_cands = data.get("candidates") or []
            candidates = [{
                "_id": c.get("_id") or c.get("id"),
                "firstName": c.get("firstName"),
                "lastName": c.get("lastName"),
                "city": c.get("city"),
                "country": c.get("country"),
                "skills": c.get("skills") or [],
                "experience": c.get("experience") or [],
                "education_level": c.get("education_level") or c.get("education"),
            } for c in body_cands]

        mdl = get_model()
        scored = []
        for cand in candidates:
            feats, matched = _prepare_features(cand, job)
            prob = float(mdl.predict_proba(feats)[:, 1][0])
            scored.append({
                "candidate_id": str(cand.get("_id")),
                "name": f'{cand.get("firstName","")} {cand.get("lastName","")}'.strip(),
                "city": cand.get("city") or cand.get("country"),
                "probability": prob,
                "matched_skills": matched,
            })
        scored.sort(key=lambda x: x["probability"], reverse=True)
        top_n = int(data.get("top_n", 5))
        return jsonify({"matches": scored[:top_n]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
