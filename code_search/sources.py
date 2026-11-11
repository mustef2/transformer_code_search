from pathlib import Path
import zipfile

MAX_FILES = 5000
MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024

def extract_uploaded_zip(archive_path: Path, destination: Path) -> Path:
    """Safely extract ZIP archives with basic traversal and size checks."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > MAX_FILES:
            raise ValueError(f"ZIP contains too many entries (limit: {MAX_FILES}).")
        if sum(m.file_size for m in members) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("ZIP exceeds the 200 MB uncompressed size limit.")
        for member in members:
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError("ZIP contains an unsafe path.")
        archive.extractall(destination)
    children = list(destination.iterdir())
    return children[0] if len(children) == 1 and children[0].is_dir() else destination
