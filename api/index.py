import os
from datetime import datetime
from flask import Flask, jsonify, request
from pymongo import MongoClient
from bson import ObjectId
import joblib
import pandas as pd
import re

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "recruitment_db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_recruitment_model.joblib")

app = Flask(__name__)
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
model = joblib.load(MODEL_PATH)

def _to_text_list(x):
    if not x: return []
    if isinstance(x, list): return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str): return [i.strip() for i in re.split(r"[|,;/]+", x) if i.strip()]
    return []

def _years_from_experience(exp_items):
    if not exp_items: return 0.0
    years = 0.0
    now = datetime.utcnow().date()
    for item in exp_items:
        sd = str(item.get("startDate") or "").split("T")[0]
        ed = str(item.get("endDate") or "").split("T")[0]
        try:
            sdate = datetime.fromisoformat(sd).date()
        except Exception:
            continue
        edate = now
        if ed:
            try:
                edate = datetime.fromisoformat(ed).date()
            except Exception:
                edate = now
        if edate > sdate:
            years += (edate - sdate).days / 365.25
    return round(max(years, 0.0), 2)

def _locations_match(cand_loc, job_loc):
    if not cand_loc or not job_loc: return 0
    c0 = str(cand_loc).lower()
    j0 = str(job_loc).lower()
    return int(any(part.strip() and part.strip() in j0 for part in re.split(r"[,/|-]", c0)))

def _parse_salary_range(s):
    if not s: return None, None, None
    cur = None
    nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", s)]
    cur_m = re.search(r"\b(usd|gbp|eur|cad|aud|inr)\b", s.lower())
    if cur_m: cur = cur_m.group(1).upper()
    if len(nums) == 1: return nums[0], nums[0], cur
    if len(nums) >= 2: return min(nums[0], nums[1]), max(nums[0], nums[1]), cur
    return None, None, cur

def _salary_match(cand_expected, job_salary):
    if cand_expected is None or job_salary is None: return 0
    js_min, js_max, _ = _parse_salary_range(job_salary)
    if js_min is None: return 0
    try:
        ce = int(str(cand_expected).replace(",", ""))
    except Exception:
        return 0
    return int(js_min <= ce <= (js_max or js_min))

def _skill_overlap(a, b):
    sa = set(_to_text_list(a))
    sb = set(_to_text_list(b))
    inter = sa & sb
    union = sa | sb
    cnt = len(inter)
    jacc = round((cnt / len(union)) if union else 0.0, 4)
    return cnt, jacc

def _prepare_features(candidate, job):
    cand_skills = _to_text_list(candidate.get("skills"))
    job_skills = _to_text_list(job.get("skills")) + _to_text_list(job.get("requiredQualifications")) + _to_text_list(job.get("preferredQualifications"))
    cand_skills_str = "|".join(cand_skills)
    job_skills_str = "|".join(job_skills)
    exp_years = _years_from_experience(candidate.get("experience"))
    edu = candidate.get("education_level") or candidate.get("education") or "Unknown"
    role = job.get("title") or "Unknown"
    industry = job.get("industry") or job.get("department") or "Unknown"
    cand_loc = candidate.get("city") or candidate.get("country") or "Unknown"
    job_loc = job.get("location") or "Unknown"
    job_type = job.get("type") or "Unknown"
    dept = job.get("department") or "Unknown"
    loc_match = _locations_match(cand_loc, job_loc)
    salary_match = _salary_match(candidate.get("expectedSalary"), job.get("salary"))
    overlap_count, jaccard = _skill_overlap(cand_skills_str, job_skills_str)
    return pd.DataFrame([{
        "candidate_skills": cand_skills_str,
        "job_skills": job_skills_str,
        "experience_years": exp_years,
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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend/<candidate_id>")
def recommend_jobs(candidate_id):
    try:
        cand = db.candidates.find_one({"_id": ObjectId(candidate_id)})
    except Exception:
        return jsonify({"error": "invalid candidate_id"}), 400
    if not cand:
        return jsonify({"error": "candidate not found"}), 404
    jobs = list(db.jobs.find({}))
    scored = []
    for job in jobs:
        feats = _prepare_features(cand, job)
        prob = float(model.predict_proba(feats)[:, 1][0])
        scored.append({
            "job_id": str(job["_id"]),
            "title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "probability": prob
        })
    scored.sort(key=lambda x: x["probability"], reverse=True)
    top_n = int(request.args.get("top_n", 5))
    return jsonify({"candidate_id": candidate_id, "recommendations": scored[:top_n]})

@app.get("/match/<job_id>")
def match_candidates(job_id):
    try:
        job = db.jobs.find_one({"_id": ObjectId(job_id)})
    except Exception:
        return jsonify({"error": "invalid job_id"}), 400
    if not job:
        return jsonify({"error": "job not found"}), 404
    cands = list(db.candidates.find({}))
    scored = []
    for cand in cands:
        feats = _prepare_features(cand, job)
        prob = float(model.predict_proba(feats)[:, 1][0])
        scored.append({
            "candidate_id": str(cand["_id"]),
            "name": f'{cand.get("firstName","")} {cand.get("lastName","")}'.strip(),
            "city": cand.get("city"),
            "probability": prob
        })
    scored.sort(key=lambda x: x["probability"], reverse=True)
    top_n = int(request.args.get("top_n", 5))
    return jsonify({"job_id": job_id, "matches": scored[:top_n]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
