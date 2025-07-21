import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load your dataset CSV or DataFrame (replace this with your real data source)

data_path = os.path.join(os.path.dirname(__file__), 'job_training.csv')
print("Loading data from:", data_path)

df = pd.read_csv(data_path)

# Assume dataset columns: candidate_skills (list or string), job_skills, education_level, job_role, experience_years, label (0 or 1)

# Preprocess skill columns: join list to string if needed
def join_skills(x):
    if isinstance(x, list):
        return " ".join(x)
    elif isinstance(x, str):
        return x
    else:
        return ""

df['candidate_skills'] = df['candidate_skills'].apply(join_skills)
df['job_skills'] = df['job_skills'].apply(join_skills)

# Label encode categorical columns
le_edu = LabelEncoder()
df['education_encoded'] = le_edu.fit_transform(df['education_level'].fillna('Unknown'))

le_role = LabelEncoder()
df['role_encoded'] = le_role.fit_transform(df['job_role'].fillna('Unknown'))

# Vectorize candidate skills and job skills separately
tfidf_candidate = TfidfVectorizer(max_features=5000)
candidate_vecs = tfidf_candidate.fit_transform(df['candidate_skills'])

tfidf_job = TfidfVectorizer(max_features=5000)
job_vecs = tfidf_job.fit_transform(df['job_skills'])

# Combine features: candidate skills + job skills + numeric features
X = np.hstack([
    candidate_vecs.toarray(),
    job_vecs.toarray(),
    df[['experience_years', 'education_encoded', 'role_encoded']].values
])

y = df['selected'].values


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save models and encoders
model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(clf, os.path.join(model_dir, 'job_match_model.pkl'))
joblib.dump(tfidf_candidate, os.path.join(model_dir, 'vectorizer_candidate.pkl'))
joblib.dump(tfidf_job, os.path.join(model_dir, 'vectorizer_job.pkl'))
joblib.dump(le_edu, os.path.join(model_dir, 'le_education.pkl'))
joblib.dump(le_role, os.path.join(model_dir, 'le_role.pkl'))

print("Training complete and models saved.")
