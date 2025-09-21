"""
Feedback generation module using LLM for personalized improvement suggestions.
"""

import json
import logging
from typing import Dict, List, Any, Optional

# Try to import Grok client
try:
    from .grok_client import grok_generate, get_client
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to reliably extract and parse a JSON object from a larger text blob.
    Handles code fences (```json ... ```), standalone { ... } blocks, and attempts
    to find a balanced JSON object by scanning braces.
    Returns parsed dict or None.
    """
    if not text or not isinstance(text, str):
        return None

    import re
    candidate = None
    # Try to find a fenced JSON block first
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
    else:
        # Fallback: find first '{' and last '}' (even if unbalanced)
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
    if not candidate:
        return None
    # Remove trailing commentary after JSON block
    candidate = re.split(r"}\s*($|\n|\r)", candidate)[0] + "}"
    # Try to parse candidate JSON; if it fails, try to fix common issues
    try:
        return json.loads(candidate)
    except Exception:
        # Remove trailing commas before } or ]
        cand2 = re.sub(r",(\s*[}\]])", r"\1", candidate)
        # Replace single quotes with double quotes
        cand2 = cand2.replace("'", '"')
        try:
            return json.loads(cand2)
        except Exception:
            return None

class FeedbackGenerator:
    """Generate personalized feedback for resume improvement."""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and GROK_AVAILABLE
        if not self.use_llm:
            logger.warning("LLM not available, using template-based feedback")
    
    def generate_one_line_feedback(self, jd_text: str, resume_text: str, missing_skills: List[str], urls: Optional[List[str]] = None) -> str:
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
            return self._generate_llm_feedback(jd_text, resume_text, missing_skills, urls=urls)
        else:
            return self._generate_template_feedback(missing_skills)
    
    def generate_detailed_feedback(self, jd_text: str, resume_text: str, missing_skills: List[str], urls: Optional[List[str]] = None) -> Dict[str, Any]:
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
            return self._generate_llm_detailed_feedback(jd_text, resume_text, missing_skills, urls=urls)
        else:
            return self._generate_template_detailed_feedback(resume_text, missing_skills)
    
    def _generate_llm_feedback(self, jd_text: str, resume_text: str, missing_skills: List[str], urls: Optional[List[str]] = None) -> str:
        """Generate feedback using LLM."""
        import re
        system_prompt = "You are an admissions coach. Provide a one-line suggestion to improve a resume given a JD and detected missing skills."
        user_prompt = f"""JD: {jd_text[:1000]}

Resume text: {resume_text[:1000]}

Missing skills: {', '.join(missing_skills)}

Produce a single, action-oriented line (imperative) that a candidate can do in 1–4 weeks to improve their fit. Example: \"Add a 2-week Kaggle project demonstrating X and host code on GitHub."""
        try:
            # Use online context when URLs provided and the client supports it
            if urls and GROK_AVAILABLE:
                client = get_client()
                response = client.generate_with_online_context(
                    prompt=user_prompt,
                    urls=urls,
                    system_prompt=system_prompt,
                    max_tokens=100,
                    temperature=0.2
                )
            else:
                response = grok_generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=100,
                    temperature=0.2
                )
            if response.get("ok"):
                raw = response["text"].strip()
                # If the model returned JSON, try to parse and extract a concise one-line
                try:
                    parsed = json.loads(raw)
                    # Prefer the first summary bullet as the one-line suggestion
                    if isinstance(parsed, dict):
                        if parsed.get("summary") and isinstance(parsed["summary"], list) and len(parsed["summary"]) > 0:
                            one_line = parsed["summary"][0]
                            return one_line
                        # fallback to use weaknesses or recommended_courses
                        if parsed.get("weaknesses"):
                            w = parsed.get("weaknesses")
                            if isinstance(w, list) and len(w) > 0:
                                return f"Focus on {w[0]} to improve your fit."
                        if parsed.get("recommended_courses"):
                            rc = parsed.get("recommended_courses")
                            if isinstance(rc, list) and len(rc) > 0:
                                return f"Recommended: {rc[0]}"
                except Exception:
                    pass
                # Clean up non-JSON response
                feedback = raw.replace('"', '').replace("'", "")
                return feedback
            else:
                logger.warning(f"LLM feedback generation failed: {response.get('error')}")
                return self._generate_template_feedback(missing_skills)
        except Exception as e:
            logger.warning(f"Error generating LLM feedback: {str(e)}")
            return self._generate_template_feedback(missing_skills)
    
    def _generate_llm_detailed_feedback(self, jd_text: str, resume_text: str, missing_skills: List[str], urls: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate detailed feedback using LLM."""
        import re
        system_prompt = "You are a resume summarizer."
        user_prompt = f"""Summarize the candidate in 2-3 short bullets focusing on relevant experience/skills for the JD.
JD: {jd_text[:2000]}

Resume: {resume_text[:2000]}

Return a JSON object with the following keys:
- "summary": list of 2-3 short bullets
- "strengths": list of strengths
- "weaknesses": list of weaknesses
- "recommended_courses": list of recommended course titles or URLs (provide URLs when possible)

Example JSON:
{{"summary": ["...","..."], "strengths": ["..."], "weaknesses": ["..."], "recommended_courses": ["https://...", "Course title - provider"]}}
"""
        try:
            if urls and GROK_AVAILABLE:
                client = get_client()
                response = client.generate_with_online_context(
                    prompt=user_prompt,
                    urls=urls,
                    system_prompt=system_prompt,
                    max_tokens=300,
                    temperature=0.2
                )
            else:
                response = grok_generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=300,
                    temperature=0.2
                )
            if response.get("ok"):
                feedback_text = response["text"].strip()
                parsed_json = _extract_json_from_text(feedback_text)
                if parsed_json:
                    recommended = parsed_json.get("recommended_courses") or parsed_json.get("recommended") or []
                    # Clean preface: remove the JSON block from the preface text
                    try:
                        m = re.search(r"```json[\s\S]*?```", feedback_text, re.IGNORECASE)
                        if m:
                            clean_preface = (feedback_text[:m.start()]).strip()
                        else:
                            first_brace = feedback_text.find('{')
                            if first_brace != -1:
                                clean_preface = feedback_text[:first_brace].strip()
                            else:
                                clean_preface = feedback_text
                    except Exception:
                        clean_preface = feedback_text
                    return {
                        "success": True,
                        "preface": clean_preface,
                        "summary": parsed_json.get("summary", []),
                        "strengths": parsed_json.get("strengths", []),
                        "weaknesses": parsed_json.get("weaknesses", []),
                        "recommended_courses": recommended,
                        "recommendation": parsed_json.get("recommendation"),
                        "interview_questions": parsed_json.get("interview_questions", []),
                        "assessment_suggestion": parsed_json.get("assessment_suggestion"),
                        "method": "llm",
                        "raw": feedback_text
                    }
                # If JSON extraction fails, show a clear error in the recruiter report
                return {
                    "success": False,
                    "preface": "AI feedback could not be parsed. Please retry or check the LLM response.",
                    "summary": [],
                    "strengths": [],
                    "weaknesses": missing_skills,
                    "recommended_courses": [],
                    "method": "llm_error",
                    "raw": feedback_text
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
    
    def generate_batch_feedback(self, resumes_data: List[Dict[str, Any]], jd_text: str, urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
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
            one_line = self.generate_one_line_feedback(jd_text, resume_text, missing_skills, urls=urls)

            # Generate detailed feedback
            detailed = self.generate_detailed_feedback(jd_text, resume_text, missing_skills, urls=urls)
            
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


def format_detailed_feedback_to_text(detailed: Dict[str, Any]) -> str:
    """
    Convert a structured detailed feedback dict into a recruiter-friendly plain text report.

    The function is resilient: if some fields are missing, it will still produce a readable report.
    """
    if not detailed or not isinstance(detailed, dict):
        return "No detailed feedback available."

    parts = []

    # Preface / opening paragraph
    def _sanitize_preface(text: Optional[str]) -> Optional[str]:
        """Remove JSON/code-fence blocks and generic instruction lines so recruiters don't see raw LLM JSON."""
        if not text:
            return None
        t = text.strip()
        # Remove fenced code blocks entirely
        import re
        t = re.sub(r"```[\s\S]*?```", "", t, flags=re.IGNORECASE).strip()
        # If what's left looks like a JSON object or starts with a JSON instruction, drop it
        if not t:
            return None
        lowered = t.lower()
        # Common instruction prefixes we want to drop
        discard_prefixes = [
            "based on",
            "based on the provided",
            "please provide",
            "here's a json",
            "here is a json",
            "json object",
            "json:",
        ]
        for p in discard_prefixes:
            if lowered.startswith(p):
                return None

        # If the string contains a JSON-like structure (lots of quotes and braces), drop it
        if ('{' in t and '}' in t) or ('"' in t and ':' in t and '[' in t):
            # likely still JSON or a fragment, so don't surface it
            return None

        # Otherwise return the cleaned preface
        return t

    preface = _sanitize_preface(detailed.get('preface'))
    if preface:
        parts.append(preface)

    # Show only LLM-generated summary and strengths if method is 'llm'
    if detailed.get('method') == 'llm':
        summary = detailed.get('summary') or []
        if summary:
            parts.append("Summary:")
            for s in summary:
                parts.append(f"- {s}")
        strengths = detailed.get('strengths') or []
        if strengths:
            parts.append("Key strengths:")
            for st in strengths:
                parts.append(f"- {st}")

    # Weaknesses / gaps
    weaknesses = detailed.get('weaknesses') or []
    if weaknesses:
        parts.append("Gaps / Areas for improvement:")
        for wk in weaknesses:
            parts.append(f"- {wk}")

    # Recommendations for recruiter (if present) or generate sensible guidance
    recs = detailed.get('recommended_recruiter_actions') or detailed.get('recommended') or []
    if recs:
        parts.append("Recommended actions for recruiter:")
        for r in recs:
            parts.append(f"- {r}")
    else:
        # Provide default, short recruiter guidance based on verdict/weaknesses only
        notes = []
        if weaknesses:
            notes.append("Consider follow-up questions or a short technical task to evaluate the weaker areas.")
        if notes:
            parts.append("Recommended actions for recruiter:")
            for n in notes:
                parts.append(f"- {n}")

    # Inline: Suggested courses (if present) - format robustly whether items are strings or dicts
    rec_courses = detailed.get('recommended_courses') or []
    if rec_courses:
        parts.append("Suggested courses:")
        for item in rec_courses:
            # Support dicts like {"title":..., "url":...} or plain strings
            if isinstance(item, dict):
                title = item.get('title') or item.get('name') or str(item)
                url = item.get('url') or item.get('link')
                if url:
                    parts.append(f"- {title} ({url})")
                else:
                    parts.append(f"- {title}")
            else:
                # Try to pull a URL out of the string
                import re
                s = str(item)
                url_match = re.search(r"(https?://\S+)", s)
                if url_match:
                    url = url_match.group(1).rstrip('.,)')
                    title = s.replace(url, '').strip(' -:;\n\t') or url
                    parts.append(f"- {title} ({url})")
                else:
                    parts.append(f"- {s}")

    # Explicit recommendation (Hire / Consider / Reject)
    recommendation = detailed.get('recommendation')
    if recommendation:
        parts.insert(0, f"Recommendation: {recommendation}")

    # Show hard match and soft match scores if present
    hard_score = detailed.get('hard_pct')
    soft_score = detailed.get('soft_pct')
    score = None
    for key in ("final_score", "score", "final_pct", "final_score_pct"):
        if key in detailed and detailed.get(key) is not None:
            score = detailed.get(key)
            break
    score_lines = []
    if hard_score is not None:
        score_lines.append(f"Hard Match Score: {hard_score:.1f}%")
    if soft_score is not None:
        score_lines.append(f"Soft Match Score: {soft_score:.1f}%")
    if score is not None:
        try:
            s = float(score)
            if 0 <= s <= 1:
                s = s * 100
            score_lines.append(f"Final Score: {s:.1f}%")
        except Exception:
            score_lines.append(f"Final Score: {score}")
    if score_lines:
        # Insert after recommendation if present, else at top
        insert_idx = 1 if recommendation else 0
        for line in reversed(score_lines):
            parts.insert(insert_idx, line)

    # Interview question suggestions
    iq = detailed.get('interview_questions') or []
    if iq:
        parts.append("Suggested interview questions:")
        for q in iq:
            parts.append(f"- {q}")

    # Short assessment suggestion (practical task or test)
    assess = detailed.get('assessment_suggestion')
    if assess:
        parts.append("Suggested quick assessment:")
        parts.append(f"- {assess}")

    # Missing skills (if provided) - helpful to show at a glance to recruiters
    missing_skills = detailed.get('missing_skills') or detailed.get('missing') or []
    if missing_skills:
        parts.append("Missing skills detected:")
        for ms in missing_skills:
            parts.append(f"- {ms}")

    # Do NOT include raw model JSON output or excerpts in the recruiter-facing report.
    # The sanitized preface above should surface any human-friendly intro only.

    # (Removed duplicate 'Suggested courses' block)
    return "\n\n".join(parts)

# Example usage

# Test: recruiter report should never show 'LLM-generated feedback' and should include suggested courses
def _test_report_sanitization():
    sample_feedback = {
        "success": True,
        "summary": ["LLM-generated feedback", "Candidate has strong Python skills"],
        "strengths": ["LLM-generated feedback", "Project experience"],
        "weaknesses": ["Spark"],
        "recommended_courses": [
            "Supervised Machine Learning: Regression and Classification - Coursera (https://www.coursera.org/specializations/machine-learning)",
            {"title": "Deep Learning", "url": "https://www.edx.org/course/deep-learning-tensorflow"}
        ],
        "recommendation": "Consider",
        "method": "llm"
    }
    report = format_detailed_feedback_to_text(sample_feedback)
    assert "LLM-generated feedback" not in report, "Report should not show 'LLM-generated feedback'"
    assert "Suggested courses" in report, "Report should include suggested courses section"
    assert "https://www.coursera.org/specializations/machine-learning" in report, "Course URL should be present"
    assert "https://www.edx.org/course/deep-learning-tensorflow" in report, "Course URL should be present"
    print("Test passed: recruiter report sanitization and course inclusion")

if __name__ == "__main__":
    # ...existing code...
    _test_report_sanitization()
