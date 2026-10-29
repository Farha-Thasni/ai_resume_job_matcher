import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄"
)
st.title("AI Resume–Job Matcher")
st.write(
    "Compare a resume with a job description "
    "using simple NLP techniques."
)
resume = st.text_area(
    "Paste your resume",
    height=220,
    placeholder="Paste your resume here..."
)
job = st.text_area(
    "Paste the job description",
    height=220,
    placeholder="Paste the job description here..."
)


def extract_keywords(text):
    words = re.findall(
        r"[a-zA-Z][a-zA-Z+#.]*",
        text.lower()
    )
    stop_words = {
        "the", "and", "for", "with", "from", "that",
        "this", "are", "you", "your", "will", "have",
        "has", "our", "into", "using", "can", "who",
        "what", "when", "where", "how", "job", "role"
    }
    return {
        word for word in words
        if len(word) > 1 and word not in stop_words
    }


if st.button("Analyze Resume", type="primary"):
    if not resume.strip() or not job.strip():
        st.warning("Please enter both texts.")
    else:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
        vectors = vectorizer.fit_transform(
            [resume, job]
        )
        score = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0] * 100
        resume_words = extract_keywords(resume)
        job_words = extract_keywords(job)
        matched = sorted(resume_words & job_words)
        missing = sorted(job_words - resume_words)
        st.subheader("Analysis Results")
        st.metric(
            "Text similarity",
            f"{score:.1f}%"
        )
        st.caption(
            "This measures textual similarity, "
            "not your actual chance of getting hired."
        )
        st.subheader("Matching keywords")
        if matched:
            st.write(", ".join(matched))
        else:
            st.write("No matching keywords found.")
        st.subheader("Keywords to review")
        if missing:
            st.write(", ".join(missing))
        else:
            st.write("No missing keywords found.")
        st.info(
            "Only add skills or experience to your resume "
            "if they accurately describe your background."
        )
