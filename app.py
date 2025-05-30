import os
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from openai import OpenAI  # Updated import
import requests
import pdfplumber
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import importlib.util
import sys

app = Flask(__name__)
load_dotenv()  # Loads OPENAI_API_KEY from .env

# Initialize Firebase Admin SDK (no need for storage since you use S3)
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)

db = firestore.client()
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Import matcher functions
try:
    from matcher import rank_resumes_by_keywords, extract_text_from_pdf
    MATCHER_AVAILABLE = True
except ImportError:
    print("Warning: matcher.py not found or contains errors. Keyword matching will not be available.")
    MATCHER_AVAILABLE = False

def get_resume_text(resume_url):
    try:
        response = requests.get(resume_url)
        if response.status_code != 200:
            print(f"Failed to download PDF: {resume_url} Status: {response.status_code}")
            return ""
        with open('temp_resume.pdf', 'wb') as f:
            f.write(response.content)
        text = ""
        with pdfplumber.open('temp_resume.pdf') as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def rank_resumes(jd_text, resumes):
    jd_emb = get_embedding(jd_text)
    scored = []
    for r in resumes:
        r_emb = get_embedding(r.get('text', ''))
        score = cosine_similarity(jd_emb, r_emb)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

@app.route('/match_resumes', methods=['POST'])
def match_resumes():
    data = request.json
    category = data.get('category', 'frontend')  # Category: frontend, backend, designers, etc.
    top_n = int(data.get('top_n', 5))
    
    # Step 1: Get candidates from the specified category
    candidates = []
    candidates_ref = db.collection('candidateCategories').document(category).collection('candidates')
    candidate_docs = candidates_ref.stream()
    
    # Dictionary to keep track of job descriptions by job_id
    job_descriptions = {}
    
    # Create a dictionary of candidates with their job_ids
    candidates_with_jobs = []
    
    for doc in candidate_docs:
        candidate_data = doc.to_dict()
        if 'name' in candidate_data and 'resume_url' in candidate_data and 'job_id' in candidate_data:
            # Store candidate with job_id
            candidates_with_jobs.append({
                'name': candidate_data['name'],
                'resume_url': candidate_data['resume_url'],
                'job_id': candidate_data['job_id'],
                'candidate_id': doc.id,
                # Preserve other fields that might be useful
                'experience': candidate_data.get('experience'),
                'email': candidate_data.get('email')
            })
    
    if not candidates_with_jobs:
        return jsonify({"error": f"No candidates found in category '{category}'"}), 404
    
    # Step 2: Get job descriptions for each unique job_id
    job_ids = set(c['job_id'] for c in candidates_with_jobs)
    jobs_ref = db.collection('jobCategories').document(category).collection('jobs')
    
    for job_id in job_ids:
        try:
            job_doc = jobs_ref.document(job_id).get()
            if job_doc.exists:
                job_data = job_doc.to_dict()
                job_descriptions[job_id] = job_data.get('job_description', '')
            else:
                print(f"Warning: Job with ID {job_id} not found")
                job_descriptions[job_id] = ""  # Set empty job description as fallback
        except Exception as e:
            print(f"Error retrieving job description for job ID {job_id}: {e}")
            job_descriptions[job_id] = ""
    
    # Step 3: Group candidates by job_id and process each group
    results = []
    
    for job_id, job_description in job_descriptions.items():
        if not job_description:
            continue  # Skip jobs with no description
            
        job_candidates = [c for c in candidates_with_jobs if c['job_id'] == job_id]
        
        # Extract text from resume URLs
        for candidate in job_candidates:
            candidate['text'] = get_resume_text(candidate['resume_url'])
        
        # Rank resumes against this specific job description
        ranked = rank_resumes(job_description, job_candidates)
        top_candidates = ranked[:top_n]
        
        # Format results for this job
        job_results = [{
            "name": r[1]['name'],
            "resume_url": r[1]['resume_url'],
            "candidate_id": r[1]['candidate_id'],
            "job_id": r[1]['job_id'],
            "score": round(r[0], 4)
        } for r in top_candidates]
        
        results.extend(job_results)
    
    # Sort all results by score (across all jobs)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Take top N overall
    final_results = results[:top_n]

    return jsonify(final_results)

if __name__ == '__main__':
    app.run(debug=True)
