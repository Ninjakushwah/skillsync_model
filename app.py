import pandas as pd
import numpy as np
from collections import Counter
import ast
import warnings
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

import pandas as pd
df = pd.read_csv("job_descriptions.csv")
print(df.shape)  # pehle size dekho
df_small = df.sample(n=500, random_state=42)  # sirf 500 rows
df_small.to_csv("job_descriptions.csv", index=False)
print("Done:", df_small.shape)


SKILLS_DB = [
    'python', 'sql', 'machine learning', 'deep learning', 'tensorflow',
    'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
    'nlp', 'computer vision', 'docker', 'git', 'aws', 'azure', 'gcp',
    'tableau', 'power bi', 'excel', 'statistics', 'data visualization',
    'feature engineering', 'neural networks', 'keras', 'spark', 'hadoop',
    'java', 'javascript', 'r', 'scala', 'mongodb', 'mysql', 'postgresql',
    'flask', 'fastapi', 'streamlit', 'linux', 'mlflow', 'airflow'
]

def parse_skills(skill_val):
    if pd.isna(skill_val):
        return []
    try:
        return ast.literal_eval(skill_val)
    except:
        return [s.strip() for s in str(skill_val).split(',')]

@st.cache_resource
def load_data_and_model():
    df = pd.read_csv("job_descriptions.csv")
    df['skills_list'] = df['skills'].apply(parse_skills)
    df['skills_str'] = df['skills_list'].apply(lambda x: ' '.join(x))
    df['job_title_clean'] = df['Job Title'].str.lower().str.strip()

    tfidf = TfidfVectorizer(max_features=100)
    skills_matrix = tfidf.fit_transform(df['skills_str'])

    le = LabelEncoder()
    df['job_label'] = le.fit_transform(df['job_title_clean'])

    X_train, X_test, y_train, y_test = train_test_split(
        skills_matrix, df['job_label'], test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return df, model, tfidf, le

def analyze_gap(df, resume_skills, job_title):
    job_data = df[df['job_title_clean'] == job_title.lower()]
    if job_data.empty:
        return None

    all_job_text = " ".join([
        str(skills) for skills in job_data['skills_list']
    ]).lower()

    required_skills = set([
        skill for skill in SKILLS_DB if skill in all_job_text
    ])

    user_skills = set([s.lower().strip() for s in resume_skills])
    matched = user_skills & required_skills
    missing = required_skills - user_skills
    match_score = round(len(matched) / len(required_skills) * 100, 2) if required_skills else 0

    return {
        "score": match_score,
        "matched": list(matched),
        "missing": list(missing)
    }

# UI
st.title("SkillSync — Job Skills Gap Analyzer")
st.markdown("Enter your skills and find out how ready you are for your target job.")

df, model, tfidf, le = load_data_and_model()

job_list = sorted(df['job_title_clean'].unique().tolist())
selected_job = st.selectbox("Select Job Title:", job_list)

user_input = st.text_area(
    "Enter your skills (comma separated):",
    placeholder="python, sql, machine learning, pandas"
)

if st.button("Analyze Gap"):
    if not user_input.strip():
        st.warning("Please enter your skills first.")
    else:
        resume_skills = [s.strip() for s in user_input.split(',')]
        result = analyze_gap(df, resume_skills, selected_job)

        if result is None:
            st.error("Job title not found in dataset.")
        else:
            st.subheader(f"Match Score: {result['score']}%")
            st.progress(int(result['score']))

            col1, col2 = st.columns(2)
            with col1:
                st.success(f"Skills you have ({len(result['matched'])}):")
                for s in result['matched']:
                    st.write(f" {s}")
            with col2:
                st.error(f"Skills you need ({len(result['missing'])}):")
                for s in result['missing']:
                    st.write(f" {s}")