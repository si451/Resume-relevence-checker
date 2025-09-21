"""
Feedback generation module using LLM for personalized improvement suggestions.
"""

import json
import logging
from typing import Dict, List, Any, Optional

# Try to import Grok client
try:
    from .grok_client import grok_generate
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class FeedbackGenerator:
    """
    Generate personalized feedback for resume improvement.
    """
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and GROK_AVAILABLE
        if not self.use_llm:
            logger.warning("LLM not available, using template-based feedback")

    def generate_one_line_feedback(self, jd_text: str, resume_text: str, missing_skills: List[str]) -> str:
        if not missing_skills:
            return "Great job! Your resume covers all the required skills."
        if self.use_llm:
            return self._generate_llm_feedback(jd_text, resume_text, missing_skills)
        else:
            return self._generate_template_feedback(missing_skills)

    def generate_detailed_feedback(self, jd_text: str, resume_text: str, missing_skills: List[str]) -> Dict[str, Any]:
        if self.use_llm:
            return self._generate_llm_detailed_feedback(jd_text, resume_text, missing_skills)
        else:
            return self._generate_template_detailed_feedback(resume_text, missing_skills)

    def _generate_llm_feedback(self, jd_text: str, resume_text: str, missing_skills: List[str]) -> str:
        system_prompt = "You are an admissions coach. Provide a one-line suggestion to improve a resume given a JD and detected missing skills."
        user_prompt = f"JD: {jd_text[:1000]}\n\nResume text: {resume_text[:1000]}\n\nMissing skills: {', '.join(missing_skills)}\n\nProduce a single, action-oriented line (imperative) that a candidate can do in 1–4 weeks to improve their fit. Example: 'Add a 2-week Kaggle project demonstrating X and host code on GitHub.'"
        try:
            response = grok_generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=100,
                temperature=0.2
            )
            if response.get("ok"):
                feedback = response["text"].strip()
                feedback = feedback.replace('"', '').replace("'", "")
                return feedback
            else:
                logger.warning(f"LLM feedback generation failed: {response.get('error')}")
                return self._generate_template_feedback(missing_skills)
        except Exception as e:
            logger.warning(f"Error generating LLM feedback: {str(e)}")
            return self._generate_template_feedback(missing_skills)

    def _generate_llm_detailed_feedback(self, jd_text: str, resume_text: str, missing_skills: List[str]) -> Dict[str, Any]:
        import re
        system_prompt = (
            "You are a resume summarizer. Always suggest at least 2 relevant online courses (with URLs) "
            "for the candidate to improve their fit, using real-time internet tools if available."
        )
        user_prompt = (
            f"Summarize the candidate in 2-3 short bullets focusing on relevant experience/skills for the JD.\n"
            f"JD: {jd_text[:1000]}\n"
            f"Resume: {resume_text[:1000]}\n"
            "Return a JSON object with the following keys:\n"
            "- 'summary': list of 2-3 short bullets\n"
            "- 'strengths': list of strengths\n"
            "- 'weaknesses': list of weaknesses\n"
            "- 'recommended_courses': list of at least 2 relevant course titles or URLs (provide URLs when possible)\n"
            "Example JSON:\n"
            "{'summary': ['...','...'], 'strengths': ['...'], 'weaknesses': ['...'], 'recommended_courses': ['https://...', 'Course title - provider']}"
        )
        try:
            response = grok_generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=300,
                temperature=0.2
            )
            if response.get("ok"):
                feedback_text = response["text"].strip()
                # Extract first valid JSON block only
                json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', feedback_text, re.DOTALL)
                if not json_blocks:
                    json_blocks = re.findall(r'(\{[\s\S]*?\})', feedback_text, re.DOTALL)
                for block in json_blocks:
                    try:
                        feedback_data = json.loads(block)
                        return {
                            "success": True,
                            "summary": feedback_data.get("summary", []),
                            "strengths": feedback_data.get("strengths", []),
                            "weaknesses": feedback_data.get("weaknesses", []),
                            "recommended_courses": feedback_data.get("recommended_courses", []),
                            "method": "llm"
                        }
                    except Exception:
                        continue
                # If no valid JSON, fallback to extracting summary lines
                summary_lines = []
                for line in feedback_text.splitlines():
                    if line.strip().startswith('- '):
                        summary_lines.append(line.strip()[2:])
                return {
                    "success": True,
                    "summary": summary_lines if summary_lines else [feedback_text],
                    "strengths": ["LLM-generated feedback"],
                    "weaknesses": missing_skills,
                    "recommended_courses": [],
                    "method": "llm_fallback"
                }
            else:
                logger.warning(f"LLM detailed feedback generation failed: {response.get('error')}")
                return self._generate_template_detailed_feedback(resume_text, missing_skills)
        except Exception as e:
            logger.warning(f"Error generating LLM detailed feedback: {str(e)}")
            return self._generate_template_detailed_feedback(resume_text, missing_skills)

    def _generate_template_feedback(self, missing_skills: List[str]) -> str:
        if not missing_skills:
            return "Great job! Your resume covers all the required skills."
        top_missing = missing_skills[:2]
        if len(top_missing) == 1:
            return f"Work on {top_missing[0]} by doing a small project and adding it to your resume."
        return f"Focus on {top_missing[0]} and {top_missing[1]} by completing relevant projects and showcasing them on GitHub."

    def _generate_template_detailed_feedback(self, resume_text: str, missing_skills: List[str]) -> Dict[str, Any]:
        resume_lower = resume_text.lower()
        has_experience = any(word in resume_lower for word in ['experience', 'worked', 'job', 'position', 'role'])
        has_education = any(word in resume_lower for word in ['education', 'degree', 'university', 'college', 'bachelor', 'master'])
        has_skills = any(word in resume_lower for word in ['skills', 'technologies', 'programming', 'languages'])
        has_projects = any(word in resume_lower for word in ['project', 'portfolio', 'github', 'repository'])
        summary = []
        if has_experience:
            summary.append("Candidate has relevant work experience")
        if has_education:
            summary.append("Educational background is present")
        if has_skills:
            summary.append("Technical skills are listed")
        if has_projects:
            summary.append("Project experience is highlighted")
        if not summary:
            summary = ["Resume needs more detailed content"]
        strengths = []
        if has_experience:
            strengths.append("Strong work experience")
        if has_education:
            strengths.append("Good educational background")
        if has_skills:
            strengths.append("Technical skills are well-documented")
        if has_projects:
            strengths.append("Project portfolio demonstrates practical skills")
        if not strengths:
            strengths = ["Resume structure is present"]
        weaknesses = missing_skills.copy()
        if not has_experience:
            weaknesses.append("Limited work experience")
        if not has_education:
            weaknesses.append("Education section could be stronger")
        if not has_skills:
            weaknesses.append("Skills section needs improvement")
        if not has_projects:
            weaknesses.append("Project portfolio is missing")
        return {
            "success": True,
            "summary": summary,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommended_courses": [],
            "method": "template"
        }

    def generate_batch_feedback(self, resumes_data: List[Dict[str, Any]], jd_text: str) -> List[Dict[str, Any]]:
        feedback_results = []
        for resume_data in resumes_data:
            resume_text = resume_data.get("text", "")
            missing_skills = resume_data.get("missing_skills", [])
            resume_file = resume_data.get("resume_file", "unknown")
            one_line = self.generate_one_line_feedback(jd_text, resume_text, missing_skills)
            detailed = self.generate_detailed_feedback(jd_text, resume_text, missing_skills)
            feedback_results.append({
                "resume_file": resume_file,
                "one_line_feedback": one_line,
                "detailed_feedback": detailed
            })
        return feedback_results

# Convenience functions
def generate_feedback(jd_text: str, resume_text: str, missing_skills: List[str], 
                     use_llm: bool = True) -> str:
    """Convenience function for generating one-line feedback."""
    generator = FeedbackGenerator(use_llm=use_llm)
    return generator.generate_one_line_feedback(jd_text, resume_text, missing_skills)

def generate_detailed_feedback(jd_text: str, resume_text: str, missing_skills: List[str], 
                              use_llm: bool = True) -> Dict[str, Any]:
    """Convenience function for generating detailed feedback."""
    generator = FeedbackGenerator(use_llm=use_llm)
    return generator.generate_detailed_feedback(jd_text, resume_text, missing_skills)

# Example usage
if __name__ == "__main__":
    # Test feedback generation
    sample_jd = "Looking for a Python developer with machine learning experience"
    sample_resume = "John Doe - Software Engineer with 3 years Python experience"
    missing_skills = ["Machine Learning", "TensorFlow"]
    
    generator = FeedbackGenerator(use_llm=False)  # Use template for testing
    
    # Test one-line feedback
    one_line = generator.generate_one_line_feedback(sample_jd, sample_resume, missing_skills)
    print(f"One-line feedback: {one_line}")
    
    # Test detailed feedback
    detailed = generator.generate_detailed_feedback(sample_jd, sample_resume, missing_skills)
    print(f"Detailed feedback: {json.dumps(detailed, indent=2)}")
