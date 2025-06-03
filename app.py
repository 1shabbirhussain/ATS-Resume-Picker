import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore # firestore module is imported
from openai import OpenAI
import requests
import pdfplumber
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import re
import datetime # Required for timezone and strftime

app = Flask(__name__)
load_dotenv()

CORS(app, resources={r"/match_resumes": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

if not firebase_admin._apps:
    try:
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", 'serviceAccountKey.json')
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialized.")
    except Exception as e:
        print(f"Error initializing Firebase Admin SDK: {e}")
db = firestore.client()

try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        print("OpenAI client initialized.")
    else:
        client = None
        print("Warning: OPENAI_API_KEY not found. OpenAI functionalities will be disabled.")
except Exception as e:
    client = None
    print(f"Error initializing OpenAI client: {e}")

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
    'javascript', 'python', 'java', 'react', 'angular', 'vue', 
    'node', 'express', 'api', 'rest', 'graphql', 'frontend', 'backend',
    'css', 'html', 'scss', 'sass', 'bootstrap', 'tailwind', 'webpack', 'npm', 'yarn',
    'typescript', 'ember', 'redux', 'jquery', 'ajax', 'typescript', 'sass', 'gulp', 
    'pwa', 'web components', 'next.js', 'gatsby', 'jquery', 'mui', 'rxjs', 'babel',
    'java', 'python', 'node.js', 'ruby', 'go', 'c#', 'c++', 'swift', 'objective-c', 'php',
    'asp.net', 'django', 'flask', 'spring', 'laravel', 'rails', 'express.js', 
    'microservices', 'serverless', 'lambda', 'graphql', 'docker', 'kubernetes', 
    'devops', 'cicd', 'jenkins', 'apache', 'nginx', 'redis', 'memcached', 'rabbitmq',
    'database', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'oracle', 'mssql', 
    'cassandra', 'redis', 'elasticsearch', 'hadoop', 'spark', 'kafka', 'dynamoDB', 
    'bigquery', 'data lake', 'data warehouse', 'etl', 'fivetran', 'airflow', 'bigdata', 
    'nosql', 'graphql', 'sqlite', 'firebase', 'realm',
    'aws', 'google cloud', 'azure', 'gcp', 'firebase', 'oracle cloud', 'ibm cloud',
    'cloud computing', 'cloud native', 'cloudformation', 'cloudwatch', 's3', 'ec2',
    'lambda', 'elastic beanstalk', 'rds', 'dynamodb', 'azure functions', 'google compute engine', 
    'kubernetes', 'containerization', 'docker', 'serverless', 'vmware', 'kubernetes', 
    'vpc', 's3', 'gke', 'eks', 'aks', 'cloud storage', 'cloud networking', 'terraform', 
    'ansible', 'chef', 'puppet', 'cloud security', 'cloud automation',
    'git', 'github', 'gitlab', 'bitbucket', 'circleci', 'jenkins', 'travis', 'azure devops', 
    'terraform', 'ansible', 'chef', 'puppet', 'bamboo', 'concourse', 'teamcity', 'flux',
    'react native', 'flutter', 'swift', 'objective-c', 'kotlin', 'java android', 'xcode', 
    'android studio', 'firebase', 'ios', 'android', 'react', 'native',
    'design', 'ux', 'ui', 'figma', 'sketch', 'photoshop', 'illustrator', 'invision', 
    'adobe xd', 'wireframes', 'prototyping', 'user interface', 'user experience',
    'testing', 'unit tests', 'integration testing', 'e2e testing', 'cypress', 'jest', 
    'mocha', 'chai', 'selenium', 'testng', 'pytest', 'jira', 'confluence', 'test automation', 
    'load testing', 'performance testing', 'bug tracking', 'continuous testing',
    'agile', 'scrum', 'kanban', 'lean', 'jira', 'confluence', 'trello', 'asana', 
    'project management', 'product management', 'devops', 'sprint', 'retrospective', 
    'user stories', 'epics', 'backlog', 'kanban board', 'product backlog',
    'oauth', 'jwt', 'soap', 'rest api', 'websocket', 'graphql', 'mqtt', 'webRTC', 'json', 
    'xml', 'swagger', 'api gateway', 'authentication', 'authorization', 'ldap', 
    'ssl', 'tls', 'tcp/ip', 'vpn', 'http', 'https', 'dns', 'rest', 'webhooks',
    'docker compose', 'microservices', 'container orchestration', 'api management', 
    'monitoring', 'logging', 'metrics', 'prometheus', 'grafana', 'kibana', 'data pipeline',
    'ci/cd', 'oauth2', 'jwt authentication', 'service mesh', 'istio', 'envoy',
    'prometheus', 'grafana', 'opentelemetry', 'openapi', 'web scraping', 'data engineering',
    'blockchain', 'cryptocurrency', 'ethereum', 'bitcoin', 'solidity', 'smart contracts', 
    'nft', 'web3', 'iot', 'devsecops', 'ai', 'machine learning', 'deep learning', 'data science', 
    'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'opencv', 'ai ethics', 
    'data visualization', 'big data', 'artificial intelligence', 'chatbot', 'computer vision',
    'nlp', 'nlp models', 'chatgpt', 'automation', 'robotics', 'neural networks', 'reinforcement learning',
    'speech recognition', 'ai models', 'self-driving', 'autonomous vehicles', "rust", "scala", "perl", "dart", "elixir", "elm", "cobol", "fortran",
    "svelte", "astro", "solid.js", "alpine.js", "litelement", "webassembly",
    "fastapi", "quarkus", "vert.x", "phoenix", "gin", "fastify",
    "clickhouse", "couchbase", "firestore", "tidb", "timescaledb", "influxdb",
    "neo4j", "snowflake", "databricks", "redshift",
    "spinnaker", "harness", "argocd", "fluxcd", "pulumi", "vagrant",
    "kubernetes operators", "helm", "skaffold", "drone ci",
    "digitalocean", "heroku", "cloudflare workers", "vercel", "netlify",
    "openstack", "linode", "ovhcloud",
    "owasp", "sast", "dast", "sonarqube", "penetration testing",
    "kali linux", "metasploit", "burp suite", "security headers",
    "content security policy",
    "jupyter", "colab", "hugging face", "apache airflow", "datarobot",
    "knime", "dask", "mlflow", "vertex ai", "automl",
    "appium", "robot framework", "testcafe", "postman", "soapui",
    "browserstack", "sauce labs", "gatling",
    "monday.com", "notion", "basecamp", "clickup", "wrike",
    "miro", "microsoft project", "smartsheet",
    "slack", "microsoft teams", "zoom", "discord", "webex", "google meet",'langchain', 'vector databases', 'pinecone', 'weaviate', 'llm', 'stable diffusion',
    'generative ai', 'prompt engineering', 'gpt-4', 'midjourney', 'llama',
    'bard', 'gemini ai', 'bedrock', 'anthropic', 'claude ai', 'azure openai',
    'chatgpt plugins', 'semantic search', 'chroma db', 'retrieval augmented generation',
    'edge computing', 'quantum computing', 'server-side rendering', 'incremental static regeneration',
    'remote procedure call', 'grpc', 'protobuf', 'webauthn', 'passkeys', 'fido2', 'zero trust',
    'event-driven architecture', 'cqrs', 'event sourcing', 'data mesh', 'feature stores',
    'reverse etl', 'vector search', 'embeddings', 'data fabric', 'digital twins'
]

def get_resume_text(resume_url):
    import io
    print(f"Downloading resume from URL: {resume_url}")
    try:
        response = requests.get(resume_url, timeout=30)
        response.raise_for_status()
        print(f"Successfully downloaded (status {response.status_code}): {resume_url}")
        
        pdf_file = io.BytesIO(response.content)
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text:
             print(f"Warning: pdfplumber extracted no text from {resume_url}. Content length: {len(response.content)}")
        else:
            print(f"Extracted text length from {resume_url}: {len(text)} characters")
        return text
    except requests.exceptions.RequestException as e:
        print(f"Requests error downloading PDF from {resume_url}: {e}")
    except pdfplumber.exceptions.PDFSyntaxError as e:
        print(f"PDFSyntaxError for {resume_url}: {e}. This might not be a valid PDF or is corrupted.")
    except Exception as e:
        print(f"Error extracting PDF text from {resume_url}: {e}")
    return ""

def get_embedding(text_input, model_name="text-embedding-3-small"):
    if not client:
        print("Error: OpenAI client not initialized. Cannot generate embeddings.")
        return []
    if not text_input or not isinstance(text_input, str) or len(text_input.strip()) == 0:
        print("Warning: Empty or invalid text provided for embedding. Returning empty list.")
        return []
    
    if len(text_input) > 30000: 
        print(f"Warning: Input text for embedding is very long ({len(text_input)} chars). Truncating to 30000 chars.")
        text_input = text_input[:30000]

    print(f"Generating embedding for text of length {len(text_input)} using model {model_name}")
    try:
        response = client.embeddings.create(
            input=text_input,
            model=model_name
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding with OpenAI: {e}")
        return []

def cosine_similarity_value(vec1, vec2):
    if not isinstance(vec1, list) or not isinstance(vec2, list) or not vec1 or not vec2 or len(vec1) != len(vec2):
        print(f"Invalid vectors for cosine similarity. vec1_len: {len(vec1) if vec1 else 0}, vec2_len: {len(vec2) if vec2 else 0}")
        return 0.0
    return 1 - cosine(vec1, vec2)

def extract_keywords(text_input):
    if not text_input: return ""
    
    text_to_process = ""
    if isinstance(text_input, list):
        text_to_process = " ".join(str(item) for item in text_input if item is not None).lower()
    elif isinstance(text_input, str):
        text_to_process = text_input.lower()
    else:
        return ""

    text_to_process = re.sub(r'[^a-zA-Z0-9\s+#.-]', ' ', text_to_process)
    
    words = [word for word in text_to_process.split() if word not in STOPWORDS and len(word) > 1]
    
    found_terms = set()
    normalized_text_for_multiword = " " + text_to_process + " " 
    for term in TECH_TERMS:
        if ' ' in term or '.' in term or '#' in term or '+' in term: 
            if f" {term} " in normalized_text_for_multiword:
                found_terms.add(term)
        elif term in words:
            found_terms.add(term)
            
    keywords = ' '.join(sorted(list(found_terms)))
    return keywords.strip()

def get_job_keywords(job_data):
    text_parts = []
    def process_field(field_data):
        if not field_data: return ""
        if isinstance(field_data, list): return " ".join(str(item) for item in field_data if item is not None)
        return str(field_data)

    text_parts.append(process_field(job_data.get('title')))
    text_parts.append(process_field(job_data.get('description')))
    text_parts.append(process_field(job_data.get('requirements')))
    text_parts.append(process_field(job_data.get('responsibilities')))
    text_parts.append(process_field(job_data.get('skills')))
    
    combined_text = " ".join(filter(None, text_parts))
    return extract_keywords(combined_text)

def rank_resumes(jd_keywords_text, resumes_with_extracted_keywords):
    if not jd_keywords_text:
        print("Warning: Job description keywords are empty.")
        return [{"keyword_score": 0.0, "candidate": r_data} for r_data in resumes_with_extracted_keywords]

    jd_keywords_emb = get_embedding(jd_keywords_text)
    if not jd_keywords_emb:
        print("Error: Could not generate embedding for job description keywords.")
        return [{"keyword_score": 0.0, "candidate": r_data} for r_data in resumes_with_extracted_keywords]

    scored_resumes = []
    print(f"Job keywords embedding dimensions: {len(jd_keywords_emb)}")
    
    for r_data in resumes_with_extracted_keywords:
        resume_keywords_text = r_data.get('keywords', '')
        score_out_of_100 = 0.0
        
        if not resume_keywords_text:
            print(f"Warning: Resume keywords are empty for candidate {r_data.get('name', 'Unknown')}.")
        else:
            resume_keywords_emb = get_embedding(resume_keywords_text)
            if not resume_keywords_emb:
                print(f"Error: Could not generate embedding for resume keywords of candidate {r_data.get('name', 'Unknown')}.")
            else:
                similarity = cosine_similarity_value(jd_keywords_emb, resume_keywords_emb)
                score_out_of_100 = round(similarity * 100, 2)
        
        scored_resumes.append({'keyword_score': score_out_of_100, 'candidate': r_data})
    
    scored_resumes.sort(key=lambda x: x['keyword_score'], reverse=True)
    print(f"Ranked {len(scored_resumes)} resumes.")
    return scored_resumes

@app.route('/match_resumes', methods=['POST'])
def match_resumes_endpoint():
    if not db: return jsonify({"error": "Firestore client not properly initialized."}), 500
    if not client and os.getenv("OPENAI_API_KEY"): return jsonify({"error": "OpenAI client not properly initialized."}), 500

    data = request.json
    category = data.get('category')
    job_id = data.get('jobId')
    top_n = int(data.get('top_n', 10))

    print(f"Received request: category={category}, jobId={job_id}, top_n={top_n}")

    if not category or not job_id:
        return jsonify({"error": "Missing 'category' (department) or 'jobId'"}), 400
    
    try:
        job_doc_ref = db.collection('jobCategories').document(category).collection('jobs').document(job_id)
        job_doc = job_doc_ref.get()
        if not job_doc.exists: return jsonify({"error": f"Job ID '{job_id}' not found in category '{category}'"}), 404
        
        job_data = job_doc.to_dict()
        job_keywords_text = get_job_keywords(job_data)
        print(f"Job Keywords: '{job_keywords_text[:200]}...'")

        candidates_ref = db.collection('candidateCategories').document(category).collection('candidates')
        query_candidates = candidates_ref.where(field_path="jobIdApplied", op_string="==", value=job_id)
        candidate_docs = query_candidates.stream()
        
        candidates_for_job_processing = []
        for doc in candidate_docs:
            candidate_data = doc.to_dict()
            if candidate_data.get('fullName') and candidate_data.get('resumeUrl'):
                submitted_at_val = candidate_data.get('submittedAt')
                submitted_at_iso_str = None

                if isinstance(submitted_at_val, str):
                    submitted_at_iso_str = submitted_at_val
                elif hasattr(submitted_at_val, 'to_datetime'): # Check for Firestore Timestamp-like object
                    try:
                        dt_obj = submitted_at_val.to_datetime(datetime.timezone.utc)
                        submitted_at_iso_str = dt_obj.strftime("%a, %d %b %Y %H:%M:%S GMT")
                    except Exception as e_format:
                        print(f"Error formatting timestamp (to_datetime): {e_format}")
                        submitted_at_iso_str = str(submitted_at_val)
                elif hasattr(submitted_at_val, 'seconds') and hasattr(submitted_at_val, 'nanoseconds'): # Fallback for other Timestamp-like objects
                    try:
                        dt_obj = datetime.datetime.fromtimestamp(submitted_at_val.seconds + submitted_at_val.nanoseconds / 1e9, tz=datetime.timezone.utc)
                        submitted_at_iso_str = dt_obj.strftime("%a, %d %b %Y %H:%M:%S GMT")
                    except Exception as e_format_secs:
                        print(f"Error formatting timestamp (seconds/nanos): {e_format_secs}")
                        submitted_at_iso_str = str(submitted_at_val)
                elif submitted_at_val is not None:
                     submitted_at_iso_str = str(submitted_at_val)
                
                candidates_for_job_processing.append({
                    'name': candidate_data['fullName'], 'resume_url': candidate_data['resumeUrl'],
                    'candidate_id': doc.id, 'email': candidate_data.get('email'),
                    'phone': candidate_data.get('phone'),
                    'experienceLevelApplied': candidate_data.get('experienceLevelApplied'),
                    'linkedIn': candidate_data.get('linkedIn'), 'portfolio': candidate_data.get('portfolio'),
                    'submittedAt': submitted_at_iso_str, 'departmentApplied': candidate_data.get('departmentApplied'),
                    'status': candidate_data.get('status', 'N/A')
                })
        print(f"Found {len(candidates_for_job_processing)} candidates for job {job_id}")

        if not candidates_for_job_processing:
             return jsonify({
                "total_candidates_analyzed": 0, "best_candidates": 0, "better_candidates": 0,
                "good_candidates": 0, "strong_match_list": [], "all_matched_list": [],
                "message": f"No candidates for job '{job_id}' in department '{category}'."
            }), 200
        
        resumes_with_keywords_for_ranking = []
        for candidate_dict in candidates_for_job_processing:
            print(f"Processing resume for: {candidate_dict['name']}")
            resume_text = get_resume_text(candidate_dict['resume_url'])
            candidate_keywords = extract_keywords(resume_text)
            resumes_with_keywords_for_ranking.append({**candidate_dict, 'keywords': candidate_keywords})
        
        ranked_candidates_scored = rank_resumes(job_keywords_text, resumes_with_keywords_for_ranking)
        
        results_list_for_response = []
        job_kw_set = set(job_keywords_text.split()) if job_keywords_text else set()

        for r_scored_item in ranked_candidates_scored:
            cand_data_item = r_scored_item['candidate']
            cand_kw_set = set(cand_data_item.get('keywords', '').split()) if cand_data_item.get('keywords') else set()
            
            results_list_for_response.append({
                "candidate_id": cand_data_item['candidate_id'],
                "departmentApplied": cand_data_item['departmentApplied'],
                "email": cand_data_item['email'],
                "experienceLevelApplied": cand_data_item['experienceLevelApplied'],
                "linkedIn": cand_data_item.get('linkedIn'),
                "matched_keywords": sorted(list(job_kw_set.intersection(cand_kw_set))),
                "name": cand_data_item['name'], 
                "phone": cand_data_item.get('phone'),
                "portfolio": cand_data_item.get('portfolio'), 
                "resume_url": cand_data_item['resume_url'],
                "score_out_of_100": r_scored_item['keyword_score'],
                "submittedAt": cand_data_item['submittedAt'], 
                "status": cand_data_item.get('status')
            })

        total_analyzed = len(results_list_for_response)
        best_c = len([r for r in results_list_for_response if r['score_out_of_100'] >= 80])
        better_c = len([r for r in results_list_for_response if 70 <= r['score_out_of_100'] < 79])
        good_c = len([r for r in results_list_for_response if 50 <= r['score_out_of_100'] < 70])
        strong_match_l = [r for r in results_list_for_response if r['score_out_of_100'] >= 75]

        return jsonify({
            "total_candidates_analyzed": total_analyzed, "best_candidates": best_c,
            "better_candidates": better_c, "good_candidates": good_c,
            "strong_match_list": strong_match_l[:top_n] if top_n and top_n > 0 and top_n < len(strong_match_l) else strong_match_l,
            "all_matched_list": results_list_for_response
        }), 200

    except Exception as e:
        import traceback
        print(f"Unhandled error: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5001)), debug=True)
