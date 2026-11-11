"""Wikipedia search and retrieval tool for agent research."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

USER_AGENT = "WikiResearchAgents/1.0 (Educational comparison bot; contact: dev@example.com)"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_REST_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"


def _http_get_json(url: str, params: dict[str, Any] | None = None, timeout: int = 15) -> dict[str, Any]:
    """Execute an HTTP GET request and return the parsed JSON response."""
    if params:
        query_string = urllib.parse.urlencode(params)
        separator = "&" if "?" in url else "?"
        full_url = f"{url}{separator}{query_string}"
    else:
        full_url = url

    req = urllib.request.Request(
        full_url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = response.getcode()
            if status_code != 200:
                raise RuntimeError(f"Wikipedia HTTP error {status_code} for URL: {full_url}")
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as http_err:
        if http_err.code == 404:
            return {"error": "not_found", "message": "Page not found (HTTP 404)"}
        raise
    except urllib.error.URLError as url_err:
        raise ConnectionError(f"Failed to connect to Wikipedia: {url_err.reason}") from url_err


def search_wikipedia_titles(query: str, limit: int = 3, timeout: int = 15) -> list[str]:
    """Search Wikipedia for titles matching the query.

    Args:
        query: Search term or subject.
        limit: Max number of matching titles to return (1-10).
        timeout: Request timeout in seconds.

    Returns:
        List of matching page titles.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    params = {
        "action": "opensearch",
        "search": clean_query,
        "limit": min(max(1, limit), 10),
        "namespace": 0,
        "format": "json",
    }

    try:
        data = _http_get_json(WIKIPEDIA_API_URL, params=params, timeout=timeout)
        # OpenSearch format: [query, [titles], [descriptions], [urls]]
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
            return data[1]
    except Exception as exc:
        logger.warning("Wikipedia title search failed for '%s': %s", clean_query, exc)

    return []


def get_wikipedia_page_summary(title: str, max_chars: int = 3000, timeout: int = 15) -> dict[str, str]:
    """Fetch the extract summary of a specific Wikipedia article title.

    Args:
        title: The exact or close title of the Wikipedia article.
        max_chars: Maximum character length of extract text.
        timeout: Request timeout in seconds.

    Returns:
        Dict with keys: 'title', 'extract', 'url', 'status' ('ok', 'not_found', 'disambiguation', 'error')
    """
    clean_title = title.strip()
    if not clean_title:
        return {"status": "error", "message": "Empty title requested."}

    encoded_title = urllib.parse.quote(clean_title.replace(" ", "_"), safe="")
    url = f"{WIKIPEDIA_REST_URL}/{encoded_title}"

    try:
        data = _http_get_json(url, timeout=timeout)
    except Exception as exc:
        return {"status": "error", "title": clean_title, "message": f"Network error: {exc}"}

    if data.get("error") == "not_found" or data.get("type") == "https://mediawiki.org/wiki/HyperSwitch/errors/not_found":
        return {
            "status": "not_found",
            "title": clean_title,
            "message": f"Wikipedia page '{clean_title}' does not exist.",
        }

    page_type = data.get("type", "standard")
    extract = data.get("extract", "")
    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

    if page_type == "disambiguation":
        return {
            "status": "disambiguation",
            "title": data.get("title", clean_title),
            "extract": extract[:max_chars],
            "url": page_url,
            "message": f"'{clean_title}' is a disambiguation page. Please specify a more targeted title.",
        }

    if not extract:
        return {
            "status": "empty",
            "title": data.get("title", clean_title),
            "message": f"Wikipedia page '{clean_title}' has no summary content.",
        }

    return {
        "status": "ok",
        "title": data.get("title", clean_title),
        "extract": extract[:max_chars],
        "url": page_url,
    }


def wikipedia_research(query: str, max_chars: int = 2500) -> str:
    """Primary tool function for researching topics on Wikipedia.

    Searches Wikipedia, retrieves the most relevant article, and returns
    factual content suitable for knowledge synthesis.

    Args:
        query: The topic or concept to research.

    Returns:
        A formatted string with factual excerpts, title, and article URL.
    """
    clean_query = query.strip()
    if not clean_query:
        return "No topic provided for Wikipedia research."

    # Step 1: Search titles
    titles = search_wikipedia_titles(clean_query, limit=3)
    if not titles:
        # Try direct lookup of query itself
        titles = [clean_query]

    # Step 2: Try fetching summary of first title, fallback to alternates if needed
    for candidate in titles:
        res = get_wikipedia_page_summary(candidate, max_chars=max_chars)
        status = res.get("status")

        if status == "ok":
            return (
                f"### Wikipedia Article: {res['title']}\n"
                f"**Source URL**: {res.get('url', 'N/A')}\n\n"
                f"{res['extract']}"
            )
        elif status == "disambiguation":
            continue

    # If all candidates failed or returned disambiguation
    first_title = titles[0] if titles else clean_query
    res = get_wikipedia_page_summary(first_title, max_chars=max_chars)
    if res.get("status") == "disambiguation":
        return (
            f"Wikipedia search for '{clean_query}' matched disambiguation topic '{res.get('title')}'.\n"
            f"Context: {res.get('extract', '')}\n"
            "Please refine the search query with a more specific subject."
        )

    return f"No detailed Wikipedia article found for topic: '{clean_query}'."

