import re
from collections import Counter

# Simple English stopwords list
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


def simple_tokenize(text):
    """Simple word tokenization"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split on whitespace
    return text.split()

def extract_keywords(text):
    """Extract keywords from text"""
    # Tokenize
    words = simple_tokenize(text)
    
    # Remove stopwords and short words
    words = [word for word in words 
             if word not in STOPWORDS 
             and len(word) > 2]  # Filter out very short words
    
    # Create simple bigrams
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append(f"{words[i]} {words[i+1]}")
    
    # Combine individual words with phrases
    all_keywords = words + bigrams
    
    # Common tech terms/skills that might be relevant
    tech_terms = [
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

    
    # Filter to include tech terms
    tech_matches = []
    for term in tech_terms:
        for keyword in all_keywords:
            if term in keyword:
                tech_matches.append(keyword)
    
    # Get the most common terms
    word_counts = Counter(all_keywords)
    common_words = [word for word, count in word_counts.most_common(30)]
    
    # Return unique keywords
    return set(common_words + tech_matches)

def calculate_keyword_match_score(resume_text, jd_text):
    """Calculate a match score based on keyword overlap"""
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)
    
    # Find matching keywords
    matched_keywords = resume_keywords.intersection(jd_keywords)
    
    # Calculate score - percentage of JD keywords found in resume
    if len(jd_keywords) == 0:
        return 0, matched_keywords, resume_keywords, jd_keywords
    
    score = len(matched_keywords) / len(jd_keywords) * 100
    return score, matched_keywords, resume_keywords, jd_keywords

def run_test_example():
    """Run a simple test example with predefined texts"""
    resume = """
     Looking for a frontend engineer proficient in React.js, JavaScript, and HTML/CSS.
    Experience in developing responsive web applications using React and Node.js is required.
    """
    jd = """
    Looking for a frontend engineer proficient in React.js, JavaScript, and HTML/CSS.
    Experience in developing responsive web applications using React and Node.js is required.
    """
    
    score, matched, resume_kw, jd_kw = calculate_keyword_match_score(resume, jd)
    
    print("\n===== TEST EXAMPLE =====")
    print(f"Resume: {resume}")
    print(f"Job Description: {jd}")
    print(f"\nResume Keywords: {resume_kw}")
    print(f"JD Keywords: {jd_kw}")
    print(f"Matched Keywords: {matched}")
    print(f"Match Score: {score:.2f}%")

if __name__ == "__main__":
    # Run the test example with hardcoded JD and resume
    run_test_example()
