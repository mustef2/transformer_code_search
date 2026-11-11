import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

def clone_public_github_repo(url: str) -> Path:
    """Shallow-clone a public GitHub repository using HTTPS."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise ValueError("Only HTTPS URLs from github.com are accepted.")
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError("Use https://github.com/owner/repository.")
    owner, repo = parts[:2]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner) or not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid GitHub owner or repository name.")
    repo = repo.removesuffix(".git")
    destination = Path(tempfile.mkdtemp(prefix="github_code_search_")) / "repository"
    result = subprocess.run(["git", "clone", "--depth", "1", "--",
                             f"https://github.com/{owner}/{repo}.git", str(destination)],
                            capture_output=True, text=True, timeout=180, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-1200:] or "Git clone failed.")
    return destination
