import re
import pdfplumber
import sklearn.metrics.pairwise as sklearn_pairwise
import spacy
import streamlit as st
from sentence_transformers import SentenceTransformer

# Load SpaCy NLP Model and Sentence Transformer
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    # Small, lightweight model perfect for beginner machines
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, embedder

nlp, embedder = load_models()

# List of target skills to look for
SKILL_BANK = [
    "python", "java", "c++", "sql", "html", "css", "javascript", "react", 
    "node.js", "docker", "aws", "git", "machine learning", "deep learning", 
    "data analysis", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "rest api", "fastapi", "flask", "communication", "leadership"
]

def extract_text_from_pdf(pdf_file):
    """Extracts raw text from an uploaded PDF file."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def extract_contact_info(text):
    """Uses Regex to find email and phone number."""
    email_pattern = r'[a-zA-Z0-9%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}'
    
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    
    return {
        "email": emails[0] if emails else "Not Found",
        "phone": phones[0] if phones else "Not Found"
    }

def extract_skills(text):
    """Identifies skills from our SKILL_BANK in the text."""
    text_lower = text.lower()
    found_skills = set()
    for skill in SKILL_BANK:
        # Check if skill exists as a standalone word/phrase
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill)
    return found_skills

def calculate_match_score(resume_text, job_description):
    """Calculates semantic similarity using AI embeddings."""
    # Convert text into numerical vectors
    embeddings = embedder.encode([resume_text, job_description])
    
    # Cosine Similarity between vector 0 (Resume) and vector 1 (Job Description)
    similarity = sklearn_pairwise.cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)

# --- STREAMLIT WEB APP INTERFACE ---
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI Automated Resume Screener & Job Matcher")
st.write("Upload a resume and paste a job description to evaluate candidate fit.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Candidate Resume")
    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

with col2:
    st.subheader("2. Target Job Description")
    job_description = st.text_area("Paste Job Description here...", height=200)

if st.button("Run Screening & Evaluation"):
    if uploaded_file is None or not job_description.strip():
        st.error("Please provide both a PDF resume and a job description.")
    else:
        with st.spinner("Analyzing resume and computing match..."):
            # Step 1: Extract Text
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # Step 2: Extract Entities & Skills
            contact_info = extract_contact_info(resume_text)
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_description)
            
            # Step 3: Compute Missing & Matched Skills
            matched_skills = resume_skills.intersection(job_skills)
            missing_skills = job_skills - resume_skills
            
            # Step 4: AI Vector Matching Score
            semantic_score = calculate_match_score(resume_text, job_description)
            final_percentage = round(semantic_score * 100, 2)
            
            # Step 5: Display Results
            st.divider()
            st.header("Evaluation Results")
            
            # Display Score Progress Bar
            st.metric(label="Overall Match Score", value=f"{final_percentage}%")
            st.progress(min(int(final_percentage), 100))
            
            # Display Contact Info
            st.subheader("Candidate Information")
            st.write(f"**Email:** {contact_info['email']}")
            st.write(f"**Phone:** {contact_info['phone']}")
            
            # Skill Gap Analysis
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("✅ Matched Skills")
                if matched_skills:
                    for s in sorted(matched_skills):
                        st.success(s)
                else:
                    st.write("No direct skill matches found.")
                    
            with col_b:
                st.subheader("❌ Missing Job Skills")
                if missing_skills:
                    for s in sorted(missing_skills):
                        st.error(s)
                else:
                    st.write("No skill gaps detected!")