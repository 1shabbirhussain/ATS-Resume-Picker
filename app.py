import os
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from openai import OpenAI  # Updated import
import requests
import pdfplumber
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from google.cloud.firestore_v1.base_query import FieldFilter  # Add this import


app = Flask(__name__)
load_dotenv()  # Loads OPENAI_API_KEY from .env

# Initialize Firebase Admin SDK (no need for storage since you use S3)
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)

db = firestore.client()
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Import matcher functions if available
try:
    from matcher import rank_resumes_by_keywords, extract_text_from_pdf
    MATCHER_AVAILABLE = True
except ImportError:
    print("Warning: matcher.py not found or contains errors. Keyword matching will not be available.")
    MATCHER_AVAILABLE = False

def get_resume_text(resume_url):
    print(f"Downloading resume from URL: {resume_url}")
    try:
        response = requests.get(resume_url)
        if response.status_code != 200:
            print(f"Failed to download PDF: {resume_url} Status: {response.status_code}")
            return ""
        print(f"Successfully downloaded PDF: {resume_url}")
        with open('temp_resume.pdf', 'wb') as f:
            f.write(response.content)
        text = ""
        with pdfplumber.open('temp_resume.pdf') as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        print(f"Extracted text length: {len(text)} characters")
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def get_embedding(text):
    print(f"Generating embedding for text of length {len(text)}")
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    print(f"Embedding response: {response}")
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def rank_resumes(jd_text, resumes):
    jd_emb = get_embedding(jd_text)
    scored = []
    print(f"Job description embedding length: {len(jd_emb)}")
    for r in resumes:
        r_emb = get_embedding(r.get('text', ''))
        score = cosine_similarity(jd_emb, r_emb)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    print(f"Ranked {len(scored)} resumes based on cosine similarity")
    return scored

def get_job_text_blob(job_data):
    parts = []
    if job_data.get('description'):
        parts.append(job_data['description'])
    if job_data.get('requirements'):
        parts.append(job_data['requirements'])
    if job_data.get('responsibilities'):
        parts.append(job_data['responsibilities'])
    if job_data.get('skills'):
        parts.append(" ".join(job_data['skills']))
    if job_data.get('benefits'):
        parts.append(" ".join(job_data['benefits']))
    combined_text = "\n".join(parts)
    print(f"Combined job text blob length: {len(combined_text)}")
    return combined_text

@app.route('/match_resumes', methods=['POST'])
def match_resumes():
    data = request.json
    category = data.get('category')
    job_id = data.get('jobId')
    top_n = int(data.get('top_n', 5))

    print(f"Received request with category: {category}, jobId: {job_id}, top_n: {top_n}")

    if not category or not job_id:
        return jsonify({"error": "Missing required parameters: 'category' and 'jobId'"}), 400
    
    # Step 1: Fetch candidates for the given category and jobId
    candidates_ref = db.collection('candidateCategories').document(category).collection('candidates')
    query = candidates_ref.where(filter=FieldFilter("jobIdApplied", "==", job_id))
    candidate_docs = query.stream()
    
    candidates_with_jobs = []
    for doc in candidate_docs:
        candidate_data = doc.to_dict()
        if 'fullName' in candidate_data and 'resumeUrl' in candidate_data:
            candidates_with_jobs.append({
                'name': candidate_data['fullName'],
                'resume_url': candidate_data['resumeUrl'],
                'job_id': candidate_data['jobIdApplied'],
                'candidate_id': doc.id,
                'experience': candidate_data.get('experience'),
                'email': candidate_data.get('email')
            })

    print(f"Found {len(candidates_with_jobs)} candidates matching criteria")

    if not candidates_with_jobs:
        return jsonify({"error": f"No candidates found for category '{category}' with jobId '{job_id}'"}), 404
    
    # Step 2: Get job data & build combined text blob
    jobs_ref = db.collection('jobCategories').document(category).collection('jobs')
    job_doc = jobs_ref.document(job_id).get()

    print(f"Job exists: {job_doc.exists}")

    if not job_doc.exists:
        return jsonify({"error": f"Job with ID '{job_id}' not found in category '{category}'"}), 404
    
    job_data = job_doc.to_dict()
    job_text_blob = get_job_text_blob(job_data)
    if not job_text_blob.strip():
        return jsonify({"error": "Job description and related fields are empty"}), 400
    
    # Step 3: Extract text from resumes and rank candidates
    for candidate in candidates_with_jobs:
        candidate['text'] = get_resume_text(candidate['resume_url'])
    
    ranked = rank_resumes(job_text_blob, candidates_with_jobs)
    top_candidates = ranked[:top_n]
    print(f"Top {top_n} candidates ranked successfully")
    
    results = [{
        "name": r[1]['name'],
        "resume_url": r[1]['resume_url'],
        "candidate_id": r[1]['candidate_id'],
        "job_id": r[1]['job_id'],
        "score": round(r[0], 4)
    } for r in top_candidates]
    print(f"Returning {len(results)} top candidates")
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
