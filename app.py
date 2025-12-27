import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import pdfplumber

# 1. Load the Secure Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="AI Career Agent", page_icon="💼", layout="wide")

st.title("🤖 Multi-Modal AI Career Agent")
st.markdown("### *RAG-Powered Resume Analysis & Job Matching*")

# --- CORE AI FUNCTIONS ---
def process_resume(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_obj:
                for page in pdf_obj.pages:
                    page_text = page.extract_text()
                    if page_text:
                        clean_page = page_text.encode("utf-8", "ignore").decode("utf-8")
                        text += clean_page + "\n"
        except Exception as e:
            st.error(f"Error reading file: {e}")
            continue
    
    # 1. Increase chunk size slightly to reduce the number of API calls
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # 2. Embedding with "Retry" logic
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_document"
    )
    
    # We will try to create the vector store. If it fails, we wait and retry.
    max_retries = 3
    for i in range(max_retries):
        try:
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            return vector_store
        except Exception as e:
            if "429" in str(e) and i < max_retries - 1:
                st.warning(f"Rate limit hit. Retrying in 30 seconds... (Attempt {i+1}/{max_retries})")
                time.sleep(30) # Wait for the quota to reset
            else:
                st.error(f"Failed after retries: {e}")
                return None

# --- SIDEBAR: DOCUMENT INGESTION ---
with st.sidebar:
    st.header("📄 Resume Database")
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True)
    if st.button("Index Resumes"):
        if uploaded_files:
            with st.spinner("Analyzing text and creating vectors..."):
                st.session_state.vector_db = process_resume(uploaded_files)
                if st.session_state.vector_db is None:
                    st.error("Failed to create vector store due to API limits. Please try again later.")
                else:
                    st.success("Indexing Complete!")
        else:
            st.error("Please upload at least one PDF.")

# --- MAIN INTERFACE: THE ANALYSIS ---
st.header("🎯 Target Job Matching")
job_description = st.text_area("Paste the Job Description (JD) here:", height=150)

if st.button("Generate Career Gap Report"):
    if "vector_db" not in st.session_state:
        st.warning("Please upload and index a resume first!")
    elif not job_description:
        st.warning("Please paste a Job Description.")
    else:
        with st.spinner("Comparing skills and calculating gaps..."):
            # RAG Retrieval: Find the parts of the resume most relevant to the JD
            relevant_docs = st.session_state.vector_db.similarity_search(job_description, k=3)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # AI Generation
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            
            prompt = f"""
            You are a Senior Technical Recruiter. Based ONLY on the resume context provided, 
            evaluate the candidate against the Job Description.
            
            RESUME CONTEXT: {context}
            ---
            JOB DESCRIPTION: {job_description}
            
            Provide:
            - **Skill Gap Table**: A table showing required skills vs. candidate evidence.
            - **The 'Red Flag' Analysis**: What are the 2 biggest reasons this candidate might fail?
            - **Optimized Bullet Point**: Rewrite one bullet point from the resume to highlight 
              skills mentioned in the JD using the 'STAR' method.
            """
            
            response = llm.invoke(prompt)
            st.markdown("---")
            st.markdown(response.content)