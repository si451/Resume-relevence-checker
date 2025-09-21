Resume Analyzer
Description
The Resume Analyzer is an AI-powered tool designed to streamline the recruitment process. It helps recruiters and hiring managers quickly assess the relevance of multiple resumes against a given job description. By leveraging both keyword matching and semantic similarity, the tool provides a comprehensive analysis and scoring for each candidate, along with actionable feedback for improvement.

Features
Job Description & Resume Parsing: Extracts text from both PDF and DOCX files.

AI-Powered Skill Extraction: Automatically identifies "must-have" and "good-to-have" skills from the job description using the Grok API.

Dual Scoring System:

Hard Match Score: Calculates a score based on the presence of essential keywords and skills in the resume.

Soft Match Score: Utilizes semantic similarity to evaluate the contextual relevance of the resume to the job description.

Weighted Final Score: Combines the hard and soft match scores based on user-defined weights to produce a final relevance score.

AI-Generated Feedback: Provides personalized, one-line feedback and detailed suggestions for each resume, including recommended online courses to bridge skill gaps.

Interactive Dashboard: A user-friendly Streamlit interface for uploading documents, configuring scoring, and visualizing results.

Detailed Analysis: Offers a comprehensive breakdown of scores, missing skills, and AI-generated feedback for each resume.

Export Results: Allows users to download the analysis results as a PDF report.

How It Works
The Resume Analyzer employs a two-pronged approach to scoring:

Hard Match (Keyword-Based): This method calculates the percentage of "must-have" skills from the job description that are explicitly mentioned in the resume. It uses regular expressions to ensure accurate matching of whole words and phrases.

Soft Match (Semantic Similarity): This advanced method goes beyond keywords to understand the underlying meaning and context of the resume and job description. It uses sentence transformers and the Grok API to generate text embeddings (vector representations) for both documents and then calculates their cosine similarity.

The final score is a weighted average of the hard and soft match scores, which can be adjusted by the user to suit their preferences.

Tech Stack
Backend: Python

Frontend: Streamlit

NLP/ML:

Sentence-Transformers

Scikit-learn

NLTK

spaCy

Grok API (for LLM-powered features)

Document Processing:

PyMuPDF

python-docx

pdfplumber

Data Handling: Pandas, NumPy

Installation
Clone the repository:

Bash

git clone https://github.com/your-username/ResumeAnalyzer.git
cd ResumeAnalyzer
Create a virtual environment and activate it:

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required dependencies:

Bash

pip install -r requirements.txt
Create a .env file in the root directory and add your Grok API key:

GROK_API_URL=https://api.groq.com/openai/v1
GROK_API_KEY="your-grok-api-key"
Usage
Run the Streamlit application:

Bash

streamlit run streamlit_app.py
Upload the Job Description:

You can either upload a PDF file or paste the job description text directly into the text area in the sidebar.

Click "Extract Skills" to let the AI identify the key skills.

Upload Resumes:

Upload one or more resumes in PDF or DOCX format.

Configure Scoring:

Adjust the weights for the hard and soft match scores using the sliders in the sidebar.

Analyze:

Click the "Analyze Resumes" button to start the analysis.

View Results:

The results will be displayed in the main area, showing the scores and a verdict for each resume.

Click on a resume to view a detailed analysis, including a scoring breakdown, missing skills, and AI-generated feedback.

Export:

You can download the detailed analysis as a PDF report.

Project Structure
ResumeAnalyzer/
├── data/
│   ├── resumes/
│   ├── jds/
│   ├── extracted_texts/
│   └── results/
├── scoring/
│   ├── __init__.py
│   ├── parser.py
│   ├── skill_extractor.py
│   ├── scoring.py
│   ├── embeddings.py
│   ├── grok_client.py
│   └── feedback.py
├── tests/
│   ├── run_tests.py
│   └── test_feedback_sim.py
├── .env
├── .gitignore
├── requirements.txt
├── streamlit_app.py
└── README.md
Contributing
Contributions are welcome! Please feel free to open an issue or submit a pull request.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request
