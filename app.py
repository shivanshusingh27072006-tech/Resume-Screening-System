import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Function to extract text from uploaded PDF resume
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

# Streamlit App UI Design
st.title("📄 AI Resume Screening & Ranking System")
st.write("Upload a Job Description and multiple candidate resumes to find the best match!")

# 1. Input Job Description
st.subheader("Step 1: Enter Job Description")
job_description = st.text_area("Paste the Job Description here:")

# 2. Upload Resumes
st.subheader("Step 2: Upload Resumes (PDF)")
uploaded_files = st.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

# 3. Process and Rank Button
if st.button("Analyze & Rank Resumes"):
    if not job_description.strip():
        st.error("Please enter a Job Description first.")
    elif not uploaded_files:
        st.error("Please upload at least one resume PDF.")
    else:
        resumes_text = []
        resume_names = []

        # Extract text from all uploaded resumes
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes_text.append(text)
            resume_names.append(file.name)

        # Combine JD and Resumes for TF-IDF Vectorization
        documents = [job_description] + resumes_text
        
        vectorizer = TfidfVectorizer().fit_transform(documents)
        vectors = vectorizer.toarray()

        # Job description vector is the first one, rest are resumes
        jd_vector = vectors[0]
        resume_vectors = vectors[1:]

        # Calculate Cosine Similarity between JD and each resume
        similarities = cosine_similarity([jd_vector], resume_vectors)[0]

        # Sort and rank resumes based on similarity score
        ranked_results = sorted(
            zip(resume_names, similarities), key=lambda x: x[1], reverse=True
        )

        # Display Results
        st.subheader("🏆 Resume Ranking Results (Best Match First)")
        for rank, (name, score) in enumerate(ranked_results, start=1):
            percentage = round(score * 100, 2)
            st.success(f"**Rank {rank}: {name}** — Match Score: **{percentage}%**")