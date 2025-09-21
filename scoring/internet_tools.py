"""Lightweight internet fetcher used to augment prompts with realtime content.

This module attempts to use LangChain's RequestsWrapper when available for robust
fetching and HTML handling. If LangChain is not installed, it falls back to
requests + BeautifulSoup. Network access is optional and failures are handled
gracefully — functions will return None on error.
"""
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    # Prefer LangChain's requests wrapper if available
    from langchain.requests import RequestsWrapper  # type: ignore
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except Exception:
    BS4_AVAILABLE = False


def fetch_text_from_url(url: str, max_chars: int = 4000) -> Optional[str]:
    """Fetch and return a cleaned text snippet from a URL.

    Returns None on any error. Result is trimmed to max_chars characters.
    """
    try:
        if LANGCHAIN_AVAILABLE:
            wrapper = RequestsWrapper()
            resp = wrapper.get(url)
            if not resp or not getattr(resp, 'text', None):
                return None
            text = resp.text
        elif BS4_AVAILABLE:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            # strip scripts/styles
            for s in soup(['script', 'style', 'noscript']):
                s.decompose()
            text = soup.get_text(separator=' ', strip=True)
        else:
            logger.warning('No HTTP/HTML client available (install langchain or requests+bs4)')
            return None

        # Basic cleanup and truncation
        cleaned = ' '.join(text.split())
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars].rsplit(' ', 1)[0] + '...'
        return cleaned

    except Exception as e:
        logger.warning(f'Failed to fetch {url}: {e}')
        return None


def fetch_and_summarize(urls: List[str], max_urls: int = 3) -> Optional[str]:
    """Fetch up to max_urls and return a tiny merged summary string.

    This is intentionally small: a concatenation of the top text snippets.
    """
    snippets = []
    for u in urls[:max_urls]:
        txt = fetch_text_from_url(u)
        if txt:
            snippets.append(txt)
    if not snippets:
        return None
    # naive summary: join snippets (could later call an LLM summarizer)
    summary = '\n\n'.join(snippets)
    return summary
