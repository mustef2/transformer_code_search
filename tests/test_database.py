import numpy as np

from code_search.database import PersistentCodeIndex


class CountingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        # Deterministic two-dimensional normalized vectors.
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def test_persistent_index_reuses_unchanged_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "sample.py"
    source.write_text("def greet(name):\n    return name\n", encoding="utf-8")
    model = CountingModel()
    index = PersistentCodeIndex(tmp_path / "index.sqlite")

    docs1, vectors1, stats1 = index.index_repository(repo, model)
    assert len(docs1) == 1
    assert vectors1.shape == (1, 2)
    assert stats1["changed_files"] == 1
    assert len(model.calls) == 1

    docs2, vectors2, stats2 = index.index_repository(repo, model)
    assert docs2 == docs1
    np.testing.assert_array_equal(vectors2, vectors1)
    assert stats2["changed_files"] == 0
    assert stats2["reused_files"] == 1
    assert len(model.calls) == 1


def test_changed_and_removed_files_are_updated(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    first = repo / "first.py"
    second = repo / "second.py"
    first.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    second.write_text("def beta():\n    return 2\n", encoding="utf-8")
    model = CountingModel()
    index = PersistentCodeIndex(tmp_path / "index.sqlite")

    docs, _, stats = index.index_repository(repo, model)
    assert len(docs) == 2
    assert stats["changed_files"] == 2

    first.write_text("def alpha():\n    return 100\n", encoding="utf-8")
    second.unlink()
    docs, _, stats = index.index_repository(repo, model)
    assert [doc["qualified_name"] for doc in docs] == ["alpha"]
    assert stats["changed_files"] == 1
    assert stats["removed_files"] == 1
    assert len(model.calls) == 2
    assert len(model.calls[1]) == 1


def test_empty_repository_returns_empty_embeddings(tmp_path):
    repo = tmp_path / "empty"
    repo.mkdir()
    docs, vectors, stats = PersistentCodeIndex(tmp_path / "index.sqlite").index_repository(
        repo, CountingModel()
    )
    assert docs == []
    assert vectors.shape == (0, 0)
    assert stats["indexed_code_units"] == 0
