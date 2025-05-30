# Resume Matcher

A simple tool for matching resumes against job descriptions based on keyword analysis.

## Features

- Extract keywords from resumes and job descriptions
- Compare keyword similarity between resumes and job descriptions
- Calculate a match score based on keyword overlap
- Support for text files and PDFs

## Usage

### Basic Usage

```bash
# Run the test example
python matcher.py --test

# Match a resume against a job description
python matcher.py --resume path/to/resume.txt --jd path/to/job_description.txt

# Match a PDF resume against a job description
python matcher.py --resume path/to/resume.pdf --jd path/to/job_description.txt
```

### Options

- `--test`: Run with a simple test example
- `--resume`: Path to a resume file (PDF or text) or direct text content
- `--jd`: Path to a job description file (PDF or text) or direct text content

## How It Works

1. The matcher extracts keywords from both the resume and job description
2. It identifies technical terms, skills, and common phrases
3. It calculates a match score based on the percentage of job description keywords found in the resume
4. It shows which keywords matched between the documents

## Integration with AV-ATS

This matcher can be used as a standalone tool or integrated with the AV-ATS application to provide an alternative matching algorithm based on keyword analysis rather than embeddings.

## Requirements

- Python 3.6+
- pdfplumber (for PDF text extraction)

## Installation

```bash
pip install pdfplumber
```
