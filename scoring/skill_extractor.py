"""
Skill extraction module for job descriptions.
Extracts must-have and good-to-have skills using LLM or rule-based methods.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Try to import LLM client
try:
    from .grok_client import grok_generate
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkillExtractor:
    """Extract skills from job descriptions using LLM or rule-based methods."""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and GROK_AVAILABLE
        if not self.use_llm:
            logger.warning("LLM not available, falling back to rule-based extraction")
    
    def extract_skills(self, jd_text: str) -> Dict[str, Any]:
        """
        Extract must-have and good-to-have skills from job description.
        
        Args:
            jd_text: Job description text
            
        Returns:
            Dictionary with extracted skills and metadata
        """
        if not jd_text or not jd_text.strip():
            return {
                "success": False,
                "error": "Empty job description text",
                "must_have": [],
                "good_to_have": [],
                "metadata": {"method": "none"}
            }
        
        try:
            if self.use_llm:
                return self._extract_with_llm(jd_text)
            else:
                return self._extract_with_rules(jd_text)
        except Exception as e:
            logger.error(f"Error extracting skills: {str(e)}")
            return {
                "success": False,
                "error": f"Extraction failed: {str(e)}",
                "must_have": [],
                "good_to_have": [],
                "metadata": {"method": "error"}
            }
    
    def _extract_with_llm(self, jd_text: str) -> Dict[str, Any]:
        """Extract skills using LLM (Grok API)."""
        system_prompt = """You are a hiring-assistant that extracts concise skill lists for resumes. Output JSON with two arrays: "must_have" and "good_to_have"."""
        
        user_prompt = f"""Given this job description, extract:
- 5 must-have skills (technical skills or qualifications that are required)
- 8 good-to-have skills (beneficial but not required)
Return output as strict JSON only.

Job Description:
{jd_text}"""
        
        try:
            response = grok_generate(
                prompt=f"{system_prompt}\n\n{user_prompt}",
                model="grok-1",
                max_tokens=512,
                temperature=0.2
            )
            
            if not response.get("ok"):
                logger.warning(f"LLM extraction failed: {response.get('error', 'Unknown error')}")
                return self._extract_with_rules(jd_text)
            
            # Parse JSON response
            llm_text = response["text"].strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', llm_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                skills_data = json.loads(json_str)
                
                must_have = skills_data.get("must_have", [])
                good_to_have = skills_data.get("good_to_have", [])
                
                # Validate and clean skills
                must_have = self._clean_skills_list(must_have)
                good_to_have = self._clean_skills_list(good_to_have)
                
                return {
                    "success": True,
                    "must_have": must_have,
                    "good_to_have": good_to_have,
                    "metadata": {
                        "method": "llm",
                        "raw_response": llm_text
                    },
                    "error": None
                }
            else:
                logger.warning("No JSON found in LLM response")
                return self._extract_with_rules(jd_text)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {str(e)}")
            return self._extract_with_rules(jd_text)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {str(e)}")
            return self._extract_with_rules(jd_text)
    
    def _extract_with_rules(self, jd_text: str) -> Dict[str, Any]:
        """Extract skills using rule-based methods."""
        # Common technical skills patterns
        technical_skills = {
            'programming': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
                'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
                'oracle', 'sqlite', 'dynamodb', 'neo4j'
            ],
            'frameworks': [
                'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express',
                'laravel', 'rails', 'asp.net', 'tensorflow', 'pytorch', 'keras',
                'scikit-learn', 'pandas', 'numpy', 'spark', 'hadoop'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
                'jenkins', 'gitlab', 'github actions'
            ],
            'tools': [
                'git', 'jira', 'confluence', 'slack', 'figma', 'tableau',
                'power bi', 'excel', 'word', 'powerpoint'
            ]
        }
        
        # Extract skills using keyword matching
        jd_lower = jd_text.lower()
        found_skills = set()
        
        for category, skills in technical_skills.items():
            for skill in skills:
                if skill in jd_lower:
                    found_skills.add(skill.title())
        
        # Additional pattern matching for common phrases
        patterns = [
            r'(?:experience with|knowledge of|proficient in|expert in)\s+([a-zA-Z\s]+?)(?:\s|,|\.|$)',
            r'(?:required|must have|essential)\s*:?\s*([a-zA-Z\s,]+?)(?:\n|$)',
            r'(?:preferred|nice to have|bonus)\s*:?\s*([a-zA-Z\s,]+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE)
            for match in matches:
                skills_in_match = [s.strip() for s in match.split(',')]
                for skill in skills_in_match:
                    if len(skill) > 2 and len(skill) < 50:
                        found_skills.add(skill.title())
        
        # Convert to lists and categorize
        all_skills = list(found_skills)
        
        # Simple heuristic: first 5 as must-have, rest as good-to-have
        must_have = all_skills[:5] if len(all_skills) >= 5 else all_skills[:3]
        good_to_have = all_skills[5:13] if len(all_skills) > 5 else all_skills[3:11]
        
        return {
            "success": True,
            "must_have": must_have,
            "good_to_have": good_to_have,
            "metadata": {
                "method": "rule_based",
                "total_found": len(all_skills)
            },
            "error": None
        }
    
    def _clean_skills_list(self, skills: List[str]) -> List[str]:
        """Clean and normalize skills list."""
        if not isinstance(skills, list):
            return []
        
        cleaned = []
        for skill in skills:
            if isinstance(skill, str):
                skill = skill.strip()
                if skill and len(skill) > 1 and len(skill) < 100:
                    # Remove common prefixes/suffixes
                    skill = re.sub(r'^(experience with|knowledge of|proficient in|expert in)\s+', '', skill, flags=re.IGNORECASE)
                    skill = re.sub(r'\s+(experience|knowledge|proficiency|expertise)$', '', skill, flags=re.IGNORECASE)
                    cleaned.append(skill.title())
        
        return list(set(cleaned))  # Remove duplicates
    
    def save_skills(self, skills_data: Dict[str, Any], jd_filename: str, output_dir: str = "data/results") -> str:
        """Save extracted skills to a JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(jd_filename).stem
        json_file = output_path / f"{base_name}_skills.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(skills_data, f, indent=2, ensure_ascii=False)
        
        return str(json_file)

def extract_skills_from_jd(jd_text: str, use_llm: bool = True) -> Dict[str, Any]:
    """Convenience function to extract skills from job description."""
    extractor = SkillExtractor(use_llm=use_llm)
    return extractor.extract_skills(jd_text)

# Example usage
if __name__ == "__main__":
    # Test with sample job description
    sample_jd = """
    We are looking for a Senior Data Scientist with the following requirements:
    
    Required Skills:
    - Python programming (3+ years)
    - Machine Learning and Deep Learning
    - SQL and database management
    - Statistical analysis
    - Data visualization
    
    Preferred Skills:
    - TensorFlow or PyTorch
    - AWS or Azure cloud platforms
    - Docker and Kubernetes
    - Git version control
    - Tableau or Power BI
    - Natural Language Processing
    - Communication skills
    """
    
    extractor = SkillExtractor(use_llm=False)  # Use rule-based for testing
    result = extractor.extract_skills(sample_jd)
    
    print("Skill Extraction Result:")
    print(f"Success: {result['success']}")
    print(f"Must-have skills: {result['must_have']}")
    print(f"Good-to-have skills: {result['good_to_have']}")
    print(f"Method: {result['metadata']['method']}")
