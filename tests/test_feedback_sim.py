"""
Integration-style test: fetch real-time online context (if network available) and run FeedbackGenerator.
This test is intentionally conservative: it won't fail CI if network or Grok keys are absent; it will print what it can fetch.
"""
import os
import json
from scoring.feedback import FeedbackGenerator
from scoring import internet_tools

# Small helper to print separators
def _p(msg):
    print('\n' + '='*10 + ' ' + msg + ' ' + '='*10 + '\n')

# Provide a sample resume + JD
resume = {
    "resume_file": "test.pdf",
    "text": "Experienced Python developer with data analysis and visualization experience. Familiar with pandas and SQL.",
    "missing_skills": ["Machine Learning"],
    "extracted_text_path": "data/extracted_texts/test.txt"
}

jd_text = "We need a Machine Learning Engineer with strong Python skills, experience in model development, and knowledge of TensorFlow or PyTorch."

# Example URLs to fetch realtime course info (replace with real URLs you want)
urls = [
    "https://www.coursera.org/learn/machine-learning",
    "https://www.edx.org/course/machine-learning"
]

_p("Attempting to fetch online context")
summary = internet_tools.fetch_and_summarize(urls)
if summary:
    print("Fetched online context summary (truncated):")
    print(summary[:1000])
else:
    print("No online content fetched (network or parser may be unavailable)")

_p("Running FeedbackGenerator")
# Use LLM if available (requires GROK_API_KEY in env), otherwise use template fallback
fg = FeedbackGenerator(use_llm=bool(os.getenv('GROK_API_KEY')))

# generate_batch_feedback expects only (resumes, jd_text) as positional args,
# so avoid passing a third positional argument here.
res = fg.generate_batch_feedback([resume], jd_text)
print(json.dumps(res, indent=2, ensure_ascii=False))

_p("Test complete")
