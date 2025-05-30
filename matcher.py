import argparse
import json
import pdfplumber
import os
import re
import string
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
    'will', 'just', 'don', 'should', 'now'
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
        'javascript', 'python', 'java', 'react', 'angular', 'vue', 
        'node', 'express', 'api', 'rest', 'graphql', 'frontend', 'backend',
        'css', 'html', 'aws', 'cloud', 'docker', 'kubernetes', 'database',
        'sql', 'nosql', 'mongodb', 'design', 'ui', 'ux', 'testing', 'agile',
        'scrum', 'mobile', 'responsive', 'seo', 'typescript', 'php', 'ruby',
        'c#', 'c++', 'swift', 'objective-c', 'flutter', 'react native',
        'django', 'flask', 'spring', 'laravel', 'rails', 'express.js', 
        'microservices', 'devops', 'cicd', 'jenkins', 'git', 'github',
        'gitlab', 'bitbucket', 'jira', 'confluence', 'serverless', 'lambda',
        'azure', 'gcp', 'firebase', 'elasticsearch', 'redis', 'postgresql',
        'mysql', 'oracle', 'mssql', 'rest api', 'soap', 'oauth', 'jwt'
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

def extract_text_from_pdf(pdf_path_or_url):
    """Extract text content from a PDF file or URL"""
    text = ""
    try:
        # Check if it's a URL
        if pdf_path_or_url.startswith('http'):
            import requests
            import tempfile
            
            # Download the PDF to a temporary file
            response = requests.get(pdf_path_or_url)
            if response.status_code != 200:
                print(f"Failed to download PDF: {pdf_path_or_url} Status: {response.status_code}")
                return ""
                
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(response.content)
            
            # Process the temp file
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # Clean up the temp file
            try:
                os.unlink(temp_path)
            except:
                pass
        else:
            # Local file
            with pdfplumber.open(pdf_path_or_url) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

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

def rank_resumes_by_keywords(jd_text, resumes):
    """
    Rank a list of resumes based on keyword matching with a job description.
    
    Args:
        jd_text (str): The job description text
        resumes (list): List of dictionaries containing resume info, with 'text' key for content
        
    Returns:
        list: Ranked list of tuples (score, resume_dict, matched_keywords)
    """
    scored = []
    
    for r in resumes:
        resume_text = r.get('text', '')
        if not resume_text:
            continue
            
        score, matched_kw, resume_kw, jd_kw = calculate_keyword_match_score(resume_text, jd_text)
        
        # Add to scored list
        scored.append((score, r, matched_kw))
    
    # Sort by score, descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

def run_test_example():
    """Run a simple test example with predefined texts"""
    resume = "Experienced frontend developer with React, JavaScript, and CSS skills."
    jd = "Looking for a frontend engineer proficient in React.js, JavaScript, and HTML/CSS."
    
    score, matched, resume_kw, jd_kw = calculate_keyword_match_score(resume, jd)
    
    print("\n===== TEST EXAMPLE =====")
    print(f"Resume: {resume}")
    print(f"Job Description: {jd}")
    print(f"\nResume Keywords: {resume_kw}")
    print(f"JD Keywords: {jd_kw}")
    print(f"Matched Keywords: {matched}")
    print(f"Match Score: {score:.2f}%")

def main():
    """Main function to run the matcher from command line"""
    parser = argparse.ArgumentParser(description="Match resume against job description using keyword extraction")
    parser.add_argument("--resume", help="Path to resume file or text content")
    parser.add_argument("--jd", help="Path to job description file or text content")
    parser.add_argument("--test", action="store_true", help="Run with test example")
    
    args = parser.parse_args()
    
    if args.test:
        run_test_example()
        return
    
    # If both resume and jd are provided
    if args.resume and args.jd:
        # Check if the inputs are files or direct text
        resume_text = ""
        jd_text = ""
        
        # Process resume input
        if os.path.isfile(args.resume):
            if args.resume.lower().endswith('.pdf'):
                resume_text = extract_text_from_pdf(args.resume)
            else:
                with open(args.resume, 'r') as f:
                    resume_text = f.read()
        else:
            resume_text = args.resume
        
        # Process job description input
        if os.path.isfile(args.jd):
            if args.jd.lower().endswith('.pdf'):
                jd_text = extract_text_from_pdf(args.jd)
            else:
                with open(args.jd, 'r') as f:
                    jd_text = f.read()
        else:
            jd_text = args.jd
        
        # Calculate match score
        score, matched, resume_kw, jd_kw = calculate_keyword_match_score(resume_text, jd_text)
        
        print("\n===== MATCH RESULTS =====")
        print(f"Resume Keywords: {resume_kw}")
        print(f"JD Keywords: {jd_kw}")
        print(f"Matched Keywords: {matched}")
        print(f"Match Score: {score:.2f}%")
    else:
        # If no arguments provided, run test example
        run_test_example()

if __name__ == "__main__":
    main()
