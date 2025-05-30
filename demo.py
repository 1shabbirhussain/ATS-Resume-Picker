from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Example job description
text = """
Chandni Raikengar
Software Engineer
Karachi, Pakistan| chandnirai.cs@gmail.com | +923060265382 | LinkedIn
Profile Summary
Motivated Software Engineer with 1 year of experience in developing dynamic user interfaces using JavaScript
frameworks. Proficient in modern web development practices, dedicated to delivering exceptional user experiences.
Seeking a challenging role to contribute to innovative projects and grow professionally.
Work Experience
Teresol Private Limited, Karachi, Pakistan July 2023 - Present
Software Design Engineer
 Working on the core banking System of Bank-AlHabib
 Working on Vue.js as a front-end developer
 Manage the UI state using XState and Node js
 Working on Finite State Machine as a middleware.
Education
 Sukkur IBA University, Sukkur, Pakistan Feb 2019 – Apr 2023
 Bachelor of Computer Science (CGPA 3.19)
 Sindh Public Higher Secondary School, Ghotki Sep 2016 – Apr 2018
 Intermediate – Pre-Engineering
Projects
Al-Habib Banking System
• Aims to develop Core Banking Software Solution for Bank Al Habib using vue js, node js and java.
• Offer essential features like Account Management, transaction processing, online banking, robust security measures,
customer relationship management.
• Provides a user-friendly interface for customers to access their accounts and perform banking activities.
• Working on the Imports module of Trade Finance.
• Develop Use cases for EPZ branch Retirement Cases includes IFDBC Pay, Contract Pay.
• Develop Use cases for CPU Branch Document Add while purchasing products.
Final Year Project: Agriculture Recommendation System
Website for farmer's ease to ask any query related to agriculture.
• A digital System for farmer's ease full-fledged website based on MERN technology.
• Java for Back-end and Java Swing for creating endpoints of semantic model.
• Separate API developed for each functionality and also admin and advisor dashboard.
E commerce Website using MERN
 A web application in which users can view and purchase products of their own choice.
 Admin can perform Crud Operations and Users can select multiple product for purchasing and add to cart.
Awards and Certificates
 Awarded National Talent Hunt Program (NTHP) scholarship from OGDCL Pakistan for 4 years Undergraduate
Program at SIBAU.
 Participated multiple Courses from Coursera, DataCamp and meta-School.
 Participated in many volunteer works.
Skills and Tools
 HTML, CSS, JavaScript, Bootstrap, React Js, Node Js, Vue Js, ES6, API Integration, Redux, MySQL, Node js,
Java, Data Structures.
 VS Code, Git lab, Jira, Postman, Insomnia, MySQL, DBeaver, GitHub, Git. 
          
"""

# Parse the text and create a tokenizer
# Specify 'english' as the language for the tokenizer
parser = PlaintextParser.from_string(text, Tokenizer("english"))

# Use LsaSummarizer
summarizer = LsaSummarizer()
summary = summarizer(parser.document, 15)  # Generate 2 sentences summary

print("Summary:")
for sentence in summary:
    print(sentence)
