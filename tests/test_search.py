import numpy as np
from code_search.search import bm25_scores, search_functions

class FakeModel:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0]], dtype=np.float32)

def documents():
    return [
        {"qualified_name": "validate_email", "name": "validate_email",
         "signature": "validate_email(address)", "docstring": "Check if email is valid",
         "source": "def validate_email(address): return '@' in address",
         "file_path": "users/validators.py", "symbol_type": "function"},
        {"qualified_name": "calculate_tax", "name": "calculate_tax",
         "signature": "calculate_tax(amount)", "docstring": "Calculate tax amount",
         "source": "def calculate_tax(amount): return amount * .15",
         "file_path": "billing/tax.py", "symbol_type": "function"},
    ]

def test_bm25_rewards_matching_terms():
    scores = bm25_scores("validate email", documents())
    assert scores[0] > scores[1]

def test_hybrid_search_returns_scores_and_explanation():
    docs = documents()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    results = search_functions("email validation", docs, vectors, FakeModel())
    assert results[0]["document"]["qualified_name"] == "validate_email"
    assert "semantic_score" in results[0] and "lexical_score" in results[0]
    assert results[0]["explanation"]

def test_filters_by_path_and_type():
    docs = documents()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    results = search_functions("tax", docs, vectors, FakeModel(),
                               file_filter="billing/*.py", symbol_type="function")
    assert len(results) == 1
    assert results[0]["document"]["file_path"] == "billing/tax.py"
