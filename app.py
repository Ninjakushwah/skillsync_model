import pandas as pd
import numpy as np
import ast
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Fixed Skills List
SKILLS_DB = [
    'python', 'sql', 'machine learning', 'deep learning', 'tensorflow',
    'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
    'nlp', 'computer vision', 'docker', 'git', 'aws', 'azure', 'gcp',
    'tableau', 'power bi', 'excel', 'statistics', 'data visualization',
    'feature engineering', 'neural networks', 'keras', 'spark', 'hadoop',
    'java', 'javascript', 'r', 'scala', 'mongodb', 'mysql', 'postgresql',
    'flask', 'fastapi', 'streamlit', 'linux', 'mlflow', 'airflow'
]

# Helper function to handle different formatting in the skills column
def parse_skills(skill_val):
    if pd.isna(skill_val):
        return []
    try:
        return ast.literal_eval(skill_val)
    except:
        return [s.strip() for s in str(skill_val).split(',')]

# Load dataset and prepare the machine learning model
def load_data_and_model():
    # Read the data file
    df = pd.read_csv("job_descriptions.csv")
    
    # Process text columns
    df['skills_list'] = df['skills'].apply(parse_skills)
    df['skills_str'] = df['skills_list'].apply(lambda x: ' '.join(x))
    df['job_title_clean'] = df['Job Title'].str.lower().str.strip()

    # Convert skills text into features using TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=100)
    skills_matrix = tfidf.fit_transform(df['skills_str'])

    # Convert textual job titles to numbers for the model
    le = LabelEncoder()
    df['job_label'] = le.fit_transform(df['job_title_clean'])

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        skills_matrix, df['job_label'], test_size=0.2, random_state=42
    )

    # Train a standard Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return df, model, tfidf, le

# Function to analyze matched and missing skills
def analyze_gap(df, resume_skills, job_title):
    job_data = df[df['job_title_clean'] == job_title.lower()]
    if job_data.empty:
        return None

    # Combine all skills text found for this specific job title
    all_job_text = " ".join([str(skills) for skills in job_data['skills_list']]).lower()

    # Find which standard skills are required for this job
    required_skills = set([skill for skill in SKILLS_DB if skill in all_job_text])

    # Clean the input user skills
    user_skills = set([s.lower().strip() for s in resume_skills])
    
    # Perform set operations to find matches and gaps
    matched = user_skills & required_skills
    missing = required_skills - user_skills
    
    # Calculate final match score percentage
    if required_skills:
        match_score = round(len(matched) / len(required_skills) * 100, 2)
    else:
        match_score = 0

    return {
        "score": match_score,
        "matched": list(matched),
        "missing": list(missing)
    }

# --- Streamlit User Interface ---

st.title("SkillSync — Job Skills Gap Analyzer")
st.markdown("Enter your skills and find out how ready you are for your target job.")

# Initialize data, model and preprocessing objects
df, model, tfidf, le = load_data_and_model()

# User input selection dropdown for unique job titles
job_list = sorted(df['job_title_clean'].unique().tolist())
selected_job = st.selectbox("Select Job Title:", job_list)

# User text input box for listing skills
user_input = st.text_area(
    "Enter your skills (comma separated):",
    placeholder="python, sql, machine learning, pandas"
)

# Execution trigger button logic
if st.button("Analyze Gap"):
    if not user_input.strip():
        st.warning("Please enter your skills first.")
    else:
        # Split input string by commas to generate list
        resume_skills = [s.strip() for s in user_input.split(',')]
        result = analyze_gap(df, resume_skills, selected_job)

        if result is None:
            st.error("Job title not found in dataset.")
        else:
            # Display results on interface
            st.subheader(f"Match Score: {result['score']}%")
            st.progress(int(result['score']))

            col1, col2 = st.columns(2)
            with col1:
                st.success(f"Skills you have ({len(result['matched'])}):")
                for s in result['matched']:
                    st.write(f"- {s}")
            with col2:
                st.error(f"Skills you need ({len(result['missing'])}):")
                for s in result['missing']:
                    st.write(f"- {s}")
