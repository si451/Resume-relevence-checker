"""
Scoring engine for resume relevance analysis.
Implements hard-match (keyword) and soft-match (semantic) scoring.
"""

import logging
import math
import re
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json

# Try to import embedding module
try:
    from .embeddings import EmbeddingManager
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeScorer:
    """Main scoring class for resume relevance analysis."""
    
    def __init__(self, hard_weight: float = 0.6, soft_weight: float = 0.4):
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        
        # Initialize embedding manager if available
        self.embedding_manager = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_manager = EmbeddingManager()
            except Exception as e:
                logger.warning(f"Failed to initialize embedding manager: {str(e)}")
                self.embedding_manager = None
    
    def calculate_hard_match_score(self, resume_text: str, must_have_skills: List[str]) -> Tuple[float, List[str]]:
        """
        Calculate hard match score based on keyword matching.
        
        Args:
            resume_text: Resume text content
            must_have_skills: List of must-have skills from JD
            
        Returns:
            Tuple of (score_percentage, missing_skills)
        """
        if not must_have_skills:
            return 100.0, []
        
        resume_lower = resume_text.lower()
        found_skills = []
        missing_skills = []
        
        for skill in must_have_skills:
            skill_lower = skill.lower()
            
            # Check for exact match
            if skill_lower in resume_lower:
                found_skills.append(skill)
            else:
                # Check for partial match (word boundaries)
                skill_words = skill_lower.split()
                if len(skill_words) == 1:
                    # Single word - check if it appears as a whole word
                    if re.search(r'\b' + re.escape(skill_lower) + r'\b', resume_lower):
                        found_skills.append(skill)
                    else:
                        missing_skills.append(skill)
                else:
                    # Multi-word skill - check if all words appear
                    if all(word in resume_lower for word in skill_words):
                        found_skills.append(skill)
                    else:
                        missing_skills.append(skill)
        
        score = (len(found_skills) / len(must_have_skills)) * 100
        return round(score, 2), missing_skills
    
    def calculate_soft_match_score(self, resume_text: str, jd_text: str) -> float:
        """
        Calculate soft match score using semantic similarity.
        
        Args:
            resume_text: Resume text content
            jd_text: Job description text content
            
        Returns:
            Similarity score (0-100)
        """
        if not self.embedding_manager:
            logger.warning("Embedding manager not available, using fallback scoring")
            return self._fallback_semantic_score(resume_text, jd_text)
        
        try:
            # Get embeddings for both texts
            resume_embedding = self.embedding_manager.get_embedding(resume_text)
            jd_embedding = self.embedding_manager.get_embedding(jd_text)
            
            if not resume_embedding or not jd_embedding:
                logger.warning("Failed to get embeddings, using fallback scoring")
                return self._fallback_semantic_score(resume_text, jd_text)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(resume_embedding, jd_embedding)
            
            # Normalize from [-1, 1] to [0, 100]
            normalized_score = ((similarity + 1) / 2) * 100
            return round(normalized_score, 2)
            
        except Exception as e:
            logger.warning(f"Semantic scoring failed: {str(e)}, using fallback")
            return self._fallback_semantic_score(resume_text, jd_text)
    
    def _fallback_semantic_score(self, resume_text: str, jd_text: str) -> float:
        """Fallback semantic scoring using TF-IDF similarity."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=1000
            )
            
            texts = [resume_text, jd_text]
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Normalize to 0-100
            normalized_score = similarity * 100
            return round(normalized_score, 2)
            
        except ImportError:
            logger.warning("scikit-learn not available, using simple word overlap")
            return self._simple_word_overlap_score(resume_text, jd_text)
        except Exception as e:
            logger.warning(f"TF-IDF scoring failed: {str(e)}, using word overlap")
            return self._simple_word_overlap_score(resume_text, jd_text)
    
    def _simple_word_overlap_score(self, resume_text: str, jd_text: str) -> float:
        """Simple word overlap scoring as final fallback."""
        import re
        
        # Extract words (alphanumeric only)
        resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
        jd_words = set(re.findall(r'\b\w+\b', jd_text.lower()))
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        resume_words = resume_words - stop_words
        jd_words = jd_words - stop_words
        
        if not jd_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(resume_words & jd_words)
        union = len(resume_words | jd_words)
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        return round(jaccard_similarity * 100, 2)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def calculate_final_score(self, hard_score: float, soft_score: float) -> float:
        """Calculate weighted final score."""
        final_score = (self.hard_weight * hard_score) + (self.soft_weight * soft_score)
        return round(final_score, 2)
    
    def get_verdict(self, final_score: float) -> str:
        """Get verdict based on final score."""
        if final_score >= 75:
            return "High"
        elif final_score >= 45:
            return "Medium"
        else:
            return "Low"
    
    def score_resume(self, resume_text: str, jd_text: str, must_have_skills: List[str], 
                    good_to_have_skills: List[str] = None) -> Dict[str, Any]:
        """
        Score a single resume against a job description.
        
        Args:
            resume_text: Resume text content
            jd_text: Job description text content
            must_have_skills: List of must-have skills
            good_to_have_skills: List of good-to-have skills (optional)
            
        Returns:
            Dictionary with scoring results
        """
        try:
            # Calculate hard match score
            hard_score, missing_skills = self.calculate_hard_match_score(resume_text, must_have_skills)
            
            # Calculate soft match score
            soft_score = self.calculate_soft_match_score(resume_text, jd_text)
            
            # Calculate final score
            final_score = self.calculate_final_score(hard_score, soft_score)
            
            # Get verdict
            verdict = self.get_verdict(final_score)
            
            return {
                "success": True,
                "hard_pct": hard_score,
                "soft_pct": soft_score,
                "final_score": final_score,
                "verdict": verdict,
                "missing_skills": missing_skills,
                "metadata": {
                    "hard_weight": self.hard_weight,
                    "soft_weight": self.soft_weight,
                    "total_must_have": len(must_have_skills),
                    "missing_count": len(missing_skills)
                }
            }
            
        except Exception as e:
            logger.error(f"Error scoring resume: {str(e)}")
            return {
                "success": False,
                "error": f"Scoring failed: {str(e)}",
                "hard_pct": 0.0,
                "soft_pct": 0.0,
                "final_score": 0.0,
                "verdict": "Low",
                "missing_skills": must_have_skills,
                "metadata": {}
            }
    
    def score_multiple_resumes(self, resumes_data: List[Dict[str, Any]], jd_text: str, 
                             must_have_skills: List[str], good_to_have_skills: List[str] = None) -> List[Dict[str, Any]]:
        """
        Score multiple resumes against a job description.
        
        Args:
            resumes_data: List of resume data dictionaries
            jd_text: Job description text
            must_have_skills: List of must-have skills
            good_to_have_skills: List of good-to-have skills (optional)
            
        Returns:
            List of scoring results
        """
        results = []
        
        for resume_data in resumes_data:
            resume_text = resume_data.get("text", "")
            resume_file = resume_data.get("filename", "unknown")
            
            if not resume_text:
                results.append({
                    "resume_file": resume_file,
                    "success": False,
                    "error": "No text content",
                    "hard_pct": 0.0,
                    "soft_pct": 0.0,
                    "final_score": 0.0,
                    "verdict": "Low",
                    "missing_skills": must_have_skills
                })
                continue
            
            # Score the resume
            score_result = self.score_resume(resume_text, jd_text, must_have_skills, good_to_have_skills)
            score_result["resume_file"] = resume_file
            results.append(score_result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str) -> str:
        """Save scoring results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return str(output_path)

# Convenience functions
def score_resume_against_jd(resume_text: str, jd_text: str, must_have_skills: List[str], 
                           hard_weight: float = 0.6, soft_weight: float = 0.4) -> Dict[str, Any]:
    """Convenience function to score a single resume."""
    scorer = ResumeScorer(hard_weight=hard_weight, soft_weight=soft_weight)
    return scorer.score_resume(resume_text, jd_text, must_have_skills)

# Example usage
if __name__ == "__main__":
    # Test scoring with sample data
    sample_resume = """
    John Doe
    Software Engineer
    
    Experience:
    - 3 years of Python development
    - Machine Learning projects using scikit-learn
    - SQL database management
    - Data visualization with matplotlib
    
    Skills:
    - Python, SQL, Machine Learning
    - Git, Docker
    - Communication skills
    """
    
    sample_jd = """
    We are looking for a Data Scientist with:
    - Python programming (required)
    - Machine Learning experience (required)
    - SQL skills (required)
    - Statistical analysis (required)
    - Data visualization (required)
    
    Nice to have:
    - TensorFlow or PyTorch
    - AWS experience
    - Docker knowledge
    """
    
    must_have_skills = ["Python", "Machine Learning", "SQL", "Statistical Analysis", "Data Visualization"]
    
    scorer = ResumeScorer()
    result = scorer.score_resume(sample_resume, sample_jd, must_have_skills)
    
    print("Scoring Result:")
    print(f"Hard Match: {result['hard_pct']}%")
    print(f"Soft Match: {result['soft_pct']}%")
    print(f"Final Score: {result['final_score']}%")
    print(f"Verdict: {result['verdict']}")
    print(f"Missing Skills: {result['missing_skills']}")
