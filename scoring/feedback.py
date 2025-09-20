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
    """Generate personalized feedback for resume improvement."""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and GROK_AVAILABLE
        if not self.use_llm:
            logger.warning("LLM not available, using template-based feedback")
    
    def generate_one_line_feedback(self, jd_text: str, resume_text: str, missing_skills: List[str]) -> str:
        """
        Generate a one-line improvement suggestion.
        
        Args:
            jd_text: Job description text
            resume_text: Resume text
            missing_skills: List of missing skills
            
        Returns:
            One-line feedback string
        """
        if not missing_skills:
            return "Great job! Your resume covers all the required skills."
        
        if self.use_llm:
            return self._generate_llm_feedback(jd_text, resume_text, missing_skills)
        else:
            return self._generate_template_feedback(missing_skills)
    
    def generate_detailed_feedback(self, jd_text: str, resume_text: str, missing_skills: List[str]) -> Dict[str, Any]:
        """
        Generate detailed feedback including summary, strengths, and weaknesses.
        
        Args:
            jd_text: Job description text
            resume_text: Resume text
            missing_skills: List of missing skills
            
        Returns:
            Dictionary with detailed feedback
        """
        if self.use_llm:
            return self._generate_llm_detailed_feedback(jd_text, resume_text, missing_skills)
        else:
            return self._generate_template_detailed_feedback(resume_text, missing_skills)
    
    def _generate_llm_feedback(self, jd_text: str, resume_text: str, missing_skills: List[str]) -> str:
        """Generate feedback using LLM."""
        system_prompt = "You are an admissions coach. Provide a one-line suggestion to improve a resume given a JD and detected missing skills."
        
        user_prompt = f"""JD: {jd_text[:1000]}

Resume text: {resume_text[:1000]}

Missing skills: {', '.join(missing_skills)}

Produce a single, action-oriented line (imperative) that a candidate can do in 1–4 weeks to improve their fit. Example: "Add a 2-week Kaggle project demonstrating X and host code on GitHub."""
        
        try:
            response = grok_generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=100,
                temperature=0.2
            )
            
            if response.get("ok"):
                feedback = response["text"].strip()
                # Clean up the response
                feedback = feedback.replace('"', '').replace("'", "")
                return feedback
            else:
                logger.warning(f"LLM feedback generation failed: {response.get('error')}")
                return self._generate_template_feedback(missing_skills)
                
        except Exception as e:
            logger.warning(f"Error generating LLM feedback: {str(e)}")
            return self._generate_template_feedback(missing_skills)
    
    def _generate_llm_detailed_feedback(self, jd_text: str, resume_text: str, missing_skills: List[str]) -> Dict[str, Any]:
        """Generate detailed feedback using LLM."""
        system_prompt = "You are a resume summarizer."
        
        user_prompt = f"""Summarize the candidate in 2-3 short bullets focusing on relevant experience/skills for the JD.
JD: {jd_text[:1000]}

Resume: {resume_text[:1000]}

Return a JSON object: {{"summary": ["...","..."], "strengths": ["..."], "weaknesses": ["..."]}}"""
        
        try:
            response = grok_generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=300,
                temperature=0.2
            )
            
            if response.get("ok"):
                feedback_text = response["text"].strip()
                
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', feedback_text, re.DOTALL)
                if json_match:
                    try:
                        feedback_data = json.loads(json_match.group())
                        return {
                            "success": True,
                            "summary": feedback_data.get("summary", []),
                            "strengths": feedback_data.get("strengths", []),
                            "weaknesses": feedback_data.get("weaknesses", []),
                            "method": "llm"
                        }
                    except json.JSONDecodeError:
                        pass
                
                # Fallback if JSON parsing fails
                return {
                    "success": True,
                    "summary": [feedback_text],
                    "strengths": ["LLM-generated feedback"],
                    "weaknesses": missing_skills,
                    "method": "llm_fallback"
                }
            else:
                logger.warning(f"LLM detailed feedback generation failed: {response.get('error')}")
                return self._generate_template_detailed_feedback(resume_text, missing_skills)
                
        except Exception as e:
            logger.warning(f"Error generating LLM detailed feedback: {str(e)}")
            return self._generate_template_detailed_feedback(resume_text, missing_skills)
    
    def _generate_template_feedback(self, missing_skills: List[str]) -> str:
        """Generate template-based feedback."""
        if not missing_skills:
            return "Great job! Your resume covers all the required skills."
        
        # Get top 2 missing skills
        top_missing = missing_skills[:2]
        
        # Generate template feedback
        if len(top_missing) == 1:
            return f"Work on {top_missing[0]} by doing a small project and adding it to your resume."
        else:
            return f"Focus on {top_missing[0]} and {top_missing[1]} by completing relevant projects and showcasing them on GitHub."
    
    def _generate_template_detailed_feedback(self, resume_text: str, missing_skills: List[str]) -> Dict[str, Any]:
        """Generate template-based detailed feedback."""
        # Simple analysis based on text content
        resume_lower = resume_text.lower()
        
        # Extract some basic information
        has_experience = any(word in resume_lower for word in ['experience', 'worked', 'job', 'position', 'role'])
        has_education = any(word in resume_lower for word in ['education', 'degree', 'university', 'college', 'bachelor', 'master'])
        has_skills = any(word in resume_lower for word in ['skills', 'technologies', 'programming', 'languages'])
        has_projects = any(word in resume_lower for word in ['project', 'portfolio', 'github', 'repository'])
        
        # Generate summary
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
        
        # Generate strengths
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
        
        # Generate weaknesses
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
            "method": "template"
        }
    
    def generate_batch_feedback(self, resumes_data: List[Dict[str, Any]], jd_text: str) -> List[Dict[str, Any]]:
        """
        Generate feedback for multiple resumes.
        
        Args:
            resumes_data: List of resume data with scoring results
            jd_text: Job description text
            
        Returns:
            List of feedback dictionaries
        """
        feedback_results = []
        
        for resume_data in resumes_data:
            resume_text = resume_data.get("text", "")
            missing_skills = resume_data.get("missing_skills", [])
            resume_file = resume_data.get("resume_file", "unknown")
            
            # Generate one-line feedback
            one_line = self.generate_one_line_feedback(jd_text, resume_text, missing_skills)
            
            # Generate detailed feedback
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
