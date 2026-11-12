import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from .parser import extract_python_functions


class PersistentCodeIndex:
    """SQLite-backed incremental index for one or more Python repositories."""

    def __init__(self, database_path="data/code_search.db"):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self):
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS repositories (
                    repo_key TEXT PRIMARY KEY,
                    root_path TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS indexed_files (
                    repo_key TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    PRIMARY KEY (repo_key, file_path),
                    FOREIGN KEY (repo_key) REFERENCES repositories(repo_key)
                        ON DELETE CASCADE
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS code_units (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_key TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    qualified_name TEXT NOT NULL,
                    document_json TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_size INTEGER NOT NULL,
                    FOREIGN KEY (repo_key, file_path)
                        REFERENCES indexed_files(repo_key, file_path)
                        ON DELETE CASCADE
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_code_units_repo ON code_units(repo_key)")

    @staticmethod
    def _file_hash(path):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _embedding_bytes(vector):
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        return array.tobytes(), int(array.size)

    def index_repository(self, repository_path, model):
        """Index new/changed files and reuse saved embeddings for unchanged files."""
        root = Path(repository_path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Repository folder does not exist: {root}")

        repo_key = str(root)
        current_files = {}
        for path in root.rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            if any(part in {
                ".git", ".venv", "venv", "__pycache__", "node_modules",
                ".mypy_cache", ".pytest_cache", "site-packages", "dist", "build"
            } for part in Path(relative).parts):
                continue
            try:
                current_files[relative] = self._file_hash(path)
            except OSError:
                continue

        with self._connect() as con:
            con.execute(
                "INSERT INTO repositories(repo_key, root_path) VALUES (?, ?) "
                "ON CONFLICT(repo_key) DO UPDATE SET root_path=excluded.root_path, "
                "updated_at=CURRENT_TIMESTAMP",
                (repo_key, str(root)),
            )
            previous = {
                row["file_path"]: row["content_hash"]
                for row in con.execute(
                    "SELECT file_path, content_hash FROM indexed_files WHERE repo_key=?",
                    (repo_key,),
                )
            }

            changed = {p for p, h in current_files.items() if previous.get(p) != h}
            removed = set(previous) - set(current_files)

            for relative in changed | removed:
                con.execute(
                    "DELETE FROM indexed_files WHERE repo_key=? AND file_path=?",
                    (repo_key, relative),
                )

            for relative in sorted(changed):
                con.execute(
                    "INSERT INTO indexed_files(repo_key, file_path, content_hash) VALUES (?, ?, ?)",
                    (repo_key, relative, current_files[relative]),
                )

        if changed:
            all_documents = extract_python_functions(root)
            new_documents = [doc for doc in all_documents if doc["file_path"] in changed]
            if new_documents:
                vectors = model.encode(
                    [doc["search_text"] for doc in new_documents],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                vectors = np.asarray(vectors, dtype=np.float32)
                if vectors.ndim != 2 or len(vectors) != len(new_documents):
                    raise ValueError("The embedding model returned an invalid embedding matrix.")
                with self._connect() as con:
                    for doc, vector in zip(new_documents, vectors):
                        blob, size = self._embedding_bytes(vector)
                        con.execute(
                            """INSERT INTO code_units
                            (repo_key, file_path, qualified_name, document_json, embedding, embedding_size)
                            VALUES (?, ?, ?, ?, ?, ?)""",
                            (repo_key, doc["file_path"], doc["qualified_name"],
                             json.dumps(doc), sqlite3.Binary(blob), size),
                        )

        with self._connect() as con:
            rows = con.execute(
                """SELECT document_json, embedding, embedding_size
                   FROM code_units WHERE repo_key=?
                   ORDER BY file_path, qualified_name""",
                (repo_key,),
            ).fetchall()

        documents = [json.loads(row["document_json"]) for row in rows]
        if not rows:
            embeddings = np.empty((0, 0), dtype=np.float32)
        else:
            sizes = {row["embedding_size"] for row in rows}
            if len(sizes) != 1:
                raise ValueError("Stored embeddings have inconsistent dimensions. Rebuild the index.")
            embeddings = np.vstack([
                np.frombuffer(row["embedding"], dtype=np.float32, count=row["embedding_size"]).copy()
                for row in rows
            ])
        return documents, embeddings, {
            "total_files": len(current_files),
            "changed_files": len(changed),
            "removed_files": len(removed),
            "reused_files": len(current_files) - len(changed),
            "indexed_code_units": len(documents),
        }

    def clear_repository(self, repository_path):
        repo_key = str(Path(repository_path).expanduser().resolve())
        with self._connect() as con:
            con.execute("DELETE FROM repositories WHERE repo_key=?", (repo_key,))
