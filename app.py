import os
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from openai import OpenAI
import requests
import pdfplumber
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from google.cloud.firestore_v1.base_query import FieldFilter
import re
from collections import Counter

app = Flask(__name__)
load_dotenv()

# Initialize Firebase Admin SDK
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)

db = firestore.client()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define stopwords and tech terms
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 
    'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 
    'ma', 'mightn', 'mustn', 'needn', 'neednt', 'shan', 'shouldn', 'wasn', 
    'weren', 'won', 'wouldn', 'was', 'weren', 'won', 'yourselves', 'yourself', 
    'd', 're', 'll', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 
    'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'neednt', 
    'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'u', 'ur', 'asap', 
    'p.m', 'a.m', 'etc', 'fyi', 'btw', 'tbh', 'omg', 'lmao', 'lol', 'idk', 
    'brb', 'np', 'tho', 'ya', 'yep', 'yeah', 'nope', 'huh', 'mm', 'eh', 'uh'
}

TECH_TERMS = [
    # Frontend Technologies
    'javascript', 'python', 'java', 'react', 'angular', 'vue', 
    'node', 'express', 'api', 'rest', 'graphql', 'frontend', 'backend',
    'css', 'html', 'scss', 'sass', 'bootstrap', 'tailwind', 'webpack', 'npm', 'yarn',
    'typescript', 'ember', 'redux', 'jquery', 'ajax', 'typescript', 'sass', 'gulp', 
    'pwa', 'web components', 'next.js', 'gatsby', 'jquery', 'mui', 'rxjs', 'babel',
    
    # Backend Technologies
    'java', 'python', 'node.js', 'ruby', 'go', 'c#', 'c++', 'swift', 'objective-c', 'php',
    'asp.net', 'django', 'flask', 'spring', 'laravel', 'rails', 'express.js', 
    'microservices', 'serverless', 'lambda', 'graphql', 'docker', 'kubernetes', 
    'devops', 'cicd', 'jenkins', 'apache', 'nginx', 'redis', 'memcached', 'rabbitmq',
    
    # Databases and Data Technologies
    'database', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'oracle', 'mssql', 
    'cassandra', 'redis', 'elasticsearch', 'hadoop', 'spark', 'kafka', 'dynamoDB', 
    'bigquery', 'data lake', 'data warehouse', 'etl', 'fivetran', 'airflow', 'bigdata', 
    'nosql', 'graphql', 'sqlite', 'firebase', 'realm',
    
    # Cloud Services & Platforms
    'aws', 'google cloud', 'azure', 'gcp', 'firebase', 'oracle cloud', 'ibm cloud',
    'cloud computing', 'cloud native', 'cloudformation', 'cloudwatch', 's3', 'ec2',
    'lambda', 'elastic beanstalk', 'rds', 'dynamodb', 'azure functions', 'google compute engine', 
    'kubernetes', 'containerization', 'docker', 'serverless', 'vmware', 'kubernetes', 
    'vpc', 's3', 'gke', 'eks', 'aks', 'cloud storage', 'cloud networking', 'terraform', 
    'ansible', 'chef', 'puppet', 'cloud security', 'cloud automation',
    
    # CI/CD Tools
    'git', 'github', 'gitlab', 'bitbucket', 'circleci', 'jenkins', 'travis', 'azure devops', 
    'terraform', 'ansible', 'chef', 'puppet', 'bamboo', 'concourse', 'teamcity', 'flux',
    
    # Mobile Development
    'react native', 'flutter', 'swift', 'objective-c', 'kotlin', 'java android', 'xcode', 
    'android studio', 'firebase', 'ios', 'android', 'react', 'native',
    
    # UX/UI Design
    'design', 'ux', 'ui', 'figma', 'sketch', 'photoshop', 'illustrator', 'invision', 
    'adobe xd', 'wireframes', 'prototyping', 'user interface', 'user experience',
    
    # Testing & QA
    'testing', 'unit tests', 'integration testing', 'e2e testing', 'cypress', 'jest', 
    'mocha', 'chai', 'selenium', 'testng', 'pytest', 'jira', 'confluence', 'test automation', 
    'load testing', 'performance testing', 'bug tracking', 'continuous testing',
    
    # Agile & Project Management
    'agile', 'scrum', 'kanban', 'lean', 'jira', 'confluence', 'trello', 'asana', 
    'project management', 'product management', 'devops', 'sprint', 'retrospective', 
    'user stories', 'epics', 'backlog', 'kanban board', 'product backlog',
    
    # Miscellaneous Technologies
    'oauth', 'jwt', 'soap', 'rest api', 'websocket', 'graphql', 'mqtt', 'webRTC', 'json', 
    'xml', 'swagger', 'api gateway', 'authentication', 'authorization', 'ldap', 
    'ssl', 'tls', 'tcp/ip', 'vpn', 'http', 'https', 'dns', 'rest', 'webhooks',
    'docker compose', 'microservices', 'container orchestration', 'api management', 
    'monitoring', 'logging', 'metrics', 'prometheus', 'grafana', 'kibana', 'data pipeline',
    'ci/cd', 'oauth2', 'jwt authentication', 'service mesh', 'istio', 'envoy',
    'prometheus', 'grafana', 'opentelemetry', 'openapi', 'web scraping', 'data engineering',
    
    # Others
    'blockchain', 'cryptocurrency', 'ethereum', 'bitcoin', 'solidity', 'smart contracts', 
    'nft', 'web3', 'iot', 'devsecops', 'ai', 'machine learning', 'deep learning', 'data science', 
    'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'opencv', 'ai ethics', 
    'data visualization', 'big data', 'artificial intelligence', 'chatbot', 'computer vision',
    'nlp', 'nlp models', 'chatgpt', 'automation', 'robotics', 'neural networks', 'reinforcement learning',
    'speech recognition', 'ai models', 'self-driving', 'autonomous vehicles'
]

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

def extract_keywords(text):
    if not text:
        return ""
    
    # If text is a list, join it into a string first
    if isinstance(text, list):
        text = " ".join(text)
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    
    # Tokenize and remove stopwords
    words = [word for word in text.split() if word not in STOPWORDS]
    
    # Find matching tech terms (including multi-word terms)
    found_terms = set()
    for term in TECH_TERMS:
        if ' ' in term:
            # Handle multi-word terms
            if term in text:
                found_terms.add(term)
        else:
            # Single word terms
            if term in words:
                found_terms.add(term)
    
    # Also include any remaining words that look like technical terms (3+ chars, no numbers)
    additional_terms = [word for word in words 
                       if len(word) >= 3 
                       and word.isalpha() 
                       and word not in STOPWORDS 
                       and word not in found_terms]
    
    # Combine and return as a single string
    keywords = ' '.join(list(found_terms)) + ' ' + ' '.join(additional_terms)
    return keywords.strip()

def get_job_keywords(job_data):
    # Extract from all relevant fields
    text_parts = []
    
    # Handle different field types
    def process_field(field):
        if not field:
            return ""
        if isinstance(field, list):
            return " ".join(str(item) for item in field)
        return str(field)
    
    if 'description' in job_data:
        text_parts.append(process_field(job_data['description']))
    if 'requirements' in job_data:
        text_parts.append(process_field(job_data['requirements']))
    if 'skills' in job_data:
        text_parts.append(process_field(job_data['skills']))
    
    combined_text = " ".join(text_parts)
    return extract_keywords(combined_text)

def rank_resumes(jd_keywords, resumes):
    # Get embeddings for job description keywords
    jd_keywords_emb = get_embedding(jd_keywords)
    
    scored = []
    print(f"Job keywords embedding length: {len(jd_keywords_emb)}")
    
    for r in resumes:
        # Keywords similarity
        r_keywords = r.get('keywords', '')
        r_keywords_emb = get_embedding(r_keywords) if r_keywords else []
        keyword_score = cosine_similarity(jd_keywords_emb, r_keywords_emb) if r_keywords_emb else 0
        
        # Calculate score out of 100 (scaled from cosine similarity)
        score_out_of_100 = round(keyword_score * 100, 2)
        
        scored.append({
            'keyword_score': score_out_of_100,
            'candidate': r
        })
    
    # Sort by score (descending)
    scored.sort(key=lambda x: x['keyword_score'], reverse=True)
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

def get_job_keywords(job_data):
    # Extract from all relevant fields
    text_parts = []
    if job_data.get('description'):
        text_parts.append(job_data['description'])
    if job_data.get('requirements'):
        text_parts.append(job_data['requirements'])
    if job_data.get('skills'):
        text_parts.append(" ".join(job_data['skills']))
    
    combined_text = " ".join(text_parts)
    return extract_keywords(combined_text)


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
    query = candidates_ref.where("jobIdApplied", "==", job_id)
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
                'email': candidate_data.get('email'),
                'phone': candidate_data.get('phone'),
                'experienceLevelApplied': candidate_data.get('experienceLevelApplied'),
                'linkedIn': candidate_data.get('linkedIn'),
                'portfolio': candidate_data.get('portfolio'),
                'submittedAt': candidate_data.get('submittedAt'),
                'departmentApplied': candidate_data.get('departmentApplied'),
            })

    print(f"Found {len(candidates_with_jobs)} candidates matching criteria")

    if not candidates_with_jobs:
        return jsonify({"error": f"No candidates found for category '{category}' with jobId '{job_id}'"}), 404
    
    # Step 2: Get job data & build combined text blob and keywords
    jobs_ref = db.collection('jobCategories').document(category).collection('jobs')
    job_doc = jobs_ref.document(job_id).get()

    print(f"Job exists: {job_doc.exists}")

    if not job_doc.exists:
        return jsonify({"error": f"Job with ID '{job_id}' not found in category '{category}'"}), 404
    
    job_data = job_doc.to_dict()
    job_keywords = get_job_keywords(job_data)
    
    # Step 3: Extract text and keywords from resumes
    for candidate in candidates_with_jobs:
        resume_text = get_resume_text(candidate['resume_url'])
        candidate['keywords'] = extract_keywords(resume_text)
    
    # Step 4: Rank candidates using both full text and keywords
    ranked = rank_resumes(job_keywords, candidates_with_jobs)
    top_candidates = ranked[:top_n]
    print(f"Top {top_n} candidates ranked successfully")
    
    # Step 5: Calculate score ranges
    score_90_100 = len([r for r in ranked if r['keyword_score'] >= 90])
    score_80_90 = len([r for r in ranked if 80 <= r['keyword_score'] < 90])
    score_50_plus = len([r for r in ranked if r['keyword_score'] >= 50])
    
    # Step 6: Prepare the response data with matched keywords
    results = [{
        "name": r['candidate']['name'],
        "resume_url": r['candidate']['resume_url'],
        "candidate_id": r['candidate']['candidate_id'],
        "job_id": r['candidate']['job_id'],
        "score_out_of_100": round(r['keyword_score'], 2),
        "matched_keywords": list(set(r['candidate']['keywords'].split()) & set(job_keywords.split())),
        "email": r['candidate']['email'],
        "phone": r['candidate']['phone'],
        "experienceLevelApplied": r['candidate']['experienceLevelApplied'],
        "linkedIn": r['candidate']['linkedIn'],
        "portfolio": r['candidate']['portfolio'],
        "submittedAt": r['candidate']['submittedAt'],
        "departmentApplied": r['candidate']['departmentApplied']
    } for r in top_candidates]
    
    # List of candidates who scored 80+
    scored_80_plus = [{
        "name": r['candidate']['name'],
        "resume_url": r['candidate']['resume_url'],
        "candidate_id": r['candidate']['candidate_id'],
        "score_out_of_100": round(r['keyword_score'], 2),
        "matched_keywords": list(set(r['candidate']['keywords'].split()) & set(job_keywords.split())),
        "email": r['candidate']['email'],
        "phone": r['candidate']['phone'],
        "experienceLevelApplied": r['candidate']['experienceLevelApplied'],
        "linkedIn": r['candidate']['linkedIn'],
        "portfolio": r['candidate']['portfolio'],
        "submittedAt": r['candidate']['submittedAt'],
        "departmentApplied": r['candidate']['departmentApplied']
    } for r in ranked if r['keyword_score'] >= 80]
    
    # List of all matched candidates
    all_matched_candidates = [{
        "name": r['candidate']['name'],
        "resume_url": r['candidate']['resume_url'],
        "candidate_id": r['candidate']['candidate_id'],
        "score_out_of_100": round(r['keyword_score'], 2),
        "matched_keywords": list(set(r['candidate']['keywords'].split()) & set(job_keywords.split())),
        "email": r['candidate']['email'],
        "phone": r['candidate']['phone'],
        "experienceLevelApplied": r['candidate']['experienceLevelApplied'],
        "linkedIn": r['candidate']['linkedIn'],
        "portfolio": r['candidate']['portfolio'],
        "submittedAt": r['candidate']['submittedAt'],
        "departmentApplied": r['candidate']['departmentApplied']
    } for r in ranked]

    print(f"Returning {len(results)} top candidates")
    return jsonify({
        "total_candidates_analyzed": len(candidates_with_jobs),
        "best_candidates": score_90_100,
        "better_candidates": score_80_90,
        "good_candidates": score_50_plus,
        "strong_match_list": scored_80_plus,
        "all_matched_list": all_matched_candidates
    })


if __name__ == '__main__':
    app.run(debug=True)
