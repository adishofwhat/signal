"""
MD5-keyed file cache for LLM responses.
Cache key = MD5(system_prompt + user_prompt).
Populating the cache locally lets the deployed app serve responses without
a live API key on every request.
"""
import hashlib
import json
from pathlib import Path

_CACHE_DIR = Path(__file__).parent.parent / "cache" / "llm_responses"


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.json"


def get_cache_key(system_prompt: str, user_prompt: str) -> str:
    content = system_prompt + user_prompt
    return hashlib.md5(content.encode()).hexdigest()


def get_cached(system_prompt: str, user_prompt: str) -> str | None:
    key = get_cache_key(system_prompt, user_prompt)
    path = _cache_path(key)
    if path.exists():
        return json.loads(path.read_text())["response"]
    return None


def save_to_cache(system_prompt: str, user_prompt: str, response: str) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = get_cache_key(system_prompt, user_prompt)
    path = _cache_path(key)
    path.write_text(json.dumps({
        "key": key,
        "system_prompt_preview": system_prompt[:300],
        "response": response,
    }, indent=2))


def cache_stats() -> dict:
    """Return count and total size of cached responses."""
    if not _CACHE_DIR.exists():
        return {"count": 0, "total_bytes": 0}
    files = list(_CACHE_DIR.glob("*.json"))
    return {
        "count": len(files),
        "total_bytes": sum(f.stat().st_size for f in files),
    }
