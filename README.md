🤖 AI Career Agent: RAG-Powered Resume Intelligence
[Live Demo Link (Paste your Streamlit URL here)]
📖 Project Overview
The AI Career Agent is a sophisticated RAG (Retrieval-Augmented Generation) system designed to bridge the gap between job seekers and hiring requirements. Unlike standard keyword scanners, this agent uses Semantic Search to understand the context of a candidate's experience and compares it against a Job Description to provide a "Brutal" Gap Analysis and AI-driven resume optimization.

🚀 Key Features
Semantic PDF Parsing: Utilizes pdfplumber and custom UTF-8 sanitization to extract clean data from complex resume layouts.

Vector Intelligence: Converts resume text into high-dimensional mathematical vectors using Google Gemini Embeddings.

Context-Aware Retrieval: Uses FAISS (Facebook AI Similarity Search) to retrieve only the most relevant parts of a resume for the LLM to analyze, reducing "hallucinations."

STAR Method Rewriting: Automatically rewrites resume bullet points using the Situation, Task, Action, Result framework to match specific job requirements.

Rate-Limit Management: Implements automated "Wait-and-Retry" logic to handle API throttling (429 errors) during large document indexing.

🛠️ Tech Stack
LLM: Google Gemini 1.5 Flash

Framework: LangChain

Vector Database: FAISS (Facebook AI Similarity Search)

Embedding Model: models/embedding-001

Parsing: pdfplumber

Deployment: Streamlit Cloud

🧠 How the RAG Pipeline Works
Ingestion: The PDF is broken down into text chunks of 1500 characters.

Vectorization: Each chunk is converted into a vector that represents its "meaning."

Indexing: Vectors are stored in a FAISS index for lightning-fast retrieval.

Querying: When a Job Description is pasted, the system finds the 3 chunks of the resume that most closely match the JD.

Generation: The LLM receives the JD and the "Relevant Chunks" to generate an accurate, data-backed report.
