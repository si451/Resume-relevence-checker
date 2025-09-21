"""
Grok API client for text generation and embeddings.
Handles API calls with retry logic and error handling.
"""

import os
import requests
import time
import json
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from . import internet_tools

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
GROK_API_URL = os.getenv("GROK_API_URL") or ""
GROK_API_KEY = os.getenv("GROK_API_KEY") or ""

# Headers for API requests
HEADERS = {
    "Authorization": f"Bearer {GROK_API_KEY}",
    "Content-Type": "application/json"
}

class GrokClient:
    """Client for interacting with Grok API."""
    
    def __init__(self, api_url: str = None, api_key: str = None):
    self.api_url = api_url if api_url is not None else GROK_API_URL
    self.api_key = api_key if api_key is not None else GROK_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Make API request with retry logic."""
    url = f"{(self.api_url or '').rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Log the request
        print(f"\n🚀 Groq API Request to {endpoint}:")
        print(f"📤 Payload: {json.dumps(payload, indent=2)}")
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    response_data = response.json() if response.content else {}
                    print(f"✅ Groq API Response (attempt {attempt + 1}):")
                    print(f"📥 Status: {response.status_code}")
                    print(f"📥 Data: {json.dumps(response_data, indent=2)}")
                    return {
                        "ok": True,
                        "text": response.text,
                        "raw": response_data
                    }
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                elif response.status_code in [500, 502, 503, 504]:  # Server errors
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error {response.status_code}, waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Groq API Error (attempt {attempt + 1}): HTTP {response.status_code}")
                    print(f"📥 Error Response: {response.text}")
                    return {
                        "ok": False,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "raw": {}
                    }
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {"ok": False, "error": "Request timeout", "raw": {}}
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return {"ok": False, "error": f"Request failed: {str(e)}", "raw": {}}
        
        return {"ok": False, "error": "Max retries exceeded", "raw": {}}
    
    def generate(self, prompt: str, model: str = "grok-1", max_tokens: int = 256, 
                temperature: float = 0.2, system_prompt: str = None) -> Dict[str, Any]:
        """
        Generate text using Grok API.
        
        Args:
            prompt: User prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            
        Returns:
            Response dictionary
        """
        # Prepare the full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = self._make_request("chat/completions", payload)
        
        if response["ok"]:
            try:
                # Try to parse JSON response
                raw_data = response["raw"]
                if isinstance(raw_data, dict):
                    # Extract text from Groq API response format
                    if "choices" in raw_data and len(raw_data["choices"]) > 0:
                        text = raw_data["choices"][0]["message"]["content"]
                    else:
                        text = (raw_data.get("text") or 
                               raw_data.get("content") or 
                               raw_data.get("response") or 
                               raw_data.get("answer") or 
                               str(raw_data))
                    response["text"] = text
                else:
                    # If raw is not a dict, use the text as is
                    response["text"] = response["text"]
            except Exception as e:
                logger.warning(f"Error parsing response: {str(e)}")
                response["text"] = response["text"]
        
        return response

    def generate_with_online_context(self, prompt: str, urls: Optional[List[str]] = None, 
                                     model: str = "grok-1", max_tokens: int = 256,
                                     temperature: float = 0.2, system_prompt: str = None) -> Dict[str, Any]:
        """Augment the prompt with online content fetched from `urls` (if provided)

        The function will try to fetch and summarize the URLs using `internet_tools`.
        If fetching fails or no URLs are provided, it falls back to a normal generate().
        """
        augmented_prompt = prompt
        try:
            if urls:
                online_summary = internet_tools.fetch_and_summarize(urls)
                if online_summary:
                    augmented_prompt = f"OnlineContext:\n{online_summary}\n\nUserPrompt:\n{prompt}"
        except Exception:
            # any failure should not block generation — fall back
            augmented_prompt = prompt

        return self.generate(augmented_prompt, model=model, max_tokens=max_tokens, 
                             temperature=temperature, system_prompt=system_prompt)
    
    def embeddings(self, texts: List[str], model: str = "text-embedding-3-small") -> Dict[str, Any]:
        """
        Get embeddings for texts using Grok API.
        
        Args:
            texts: List of texts to embed
            model: Embedding model name
            
        Returns:
            Response dictionary with embeddings
        """
        payload = {
            "model": model,
            "input": texts
        }
        
        response = self._make_request("embeddings", payload)
        
        if response["ok"]:
            try:
                raw_data = response["raw"]
                if isinstance(raw_data, dict):
                    if "data" in raw_data:
                        embeddings = [item["embedding"] for item in raw_data["data"]]
                    else:
                        embeddings = raw_data.get("embeddings", [])
                    response["embeddings"] = embeddings
                else:
                    response["embeddings"] = []
            except Exception as e:
                logger.warning(f"Error parsing embeddings response: {str(e)}")
                response["embeddings"] = []
        
        return response
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is accessible."""
        try:
            # Simple health check - try to generate a short response
            response = self.generate("Hello", max_tokens=5)
            return {
                "ok": response["ok"],
                "status": "healthy" if response["ok"] else "unhealthy",
                "error": response.get("error")
            }
        except Exception as e:
            return {
                "ok": False,
                "status": "unhealthy",
                "error": str(e)
            }

# Global client instance
_client = None

def get_client() -> GrokClient:
    """Get or create global Grok client instance."""
    global _client
    if _client is None:
        _client = GrokClient()
    return _client

# Convenience functions
def grok_generate(prompt: str, model: str = "grok-1", max_tokens: int = 256, 
                 temperature: float = 0.2, system_prompt: str = None) -> Dict[str, Any]:
    """Convenience function for text generation."""
    client = get_client()
    return client.generate(prompt, model, max_tokens, temperature, system_prompt)

def grok_embeddings(texts: List[str], model: str = "grok-embeddings") -> Dict[str, Any]:
    """Convenience function for getting embeddings."""
    client = get_client()
    return client.embeddings(texts, model)

def check_grok_health() -> Dict[str, Any]:
    """Check if Grok API is healthy."""
    client = get_client()
    return client.health_check()

# Example usage
if __name__ == "__main__":
    # Test the client
    print("Testing Grok API client...")
    
    # Check health
    health = check_grok_health()
    print(f"Health check: {health}")
    
    if health["ok"]:
        # Test text generation
        response = grok_generate("What is Python?", max_tokens=50)
        print(f"Generation response: {response}")
        
        # Test embeddings
        embeddings_response = grok_embeddings(["Python programming", "Machine learning"])
        print(f"Embeddings response: {embeddings_response}")
    else:
        print("API is not healthy, skipping tests")
        print("Make sure to set GROK_API_URL and GROK_API_KEY in your .env file")
