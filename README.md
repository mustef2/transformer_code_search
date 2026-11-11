# AI-Powered Code Search Using Transformer Embeddings

Python application for natural-language search over local Python projects, uploaded ZIP archives, and public GitHub repositories.

## Phase 2 features

- Hybrid ranking combines Transformer semantic similarity and BM25-style lexical relevance.
- Search weight slider controls the semantic/lexical balance.
- File/folder glob filtering and code-unit filtering (functions, methods, async functions/methods).
- Results show combined, semantic and lexical scores, source locations, code previews and ranking explanations.
- Python AST extraction includes decorated functions and async functions.

## Architecture

1. Streamlit user interface.
2. Python AST parser extracts code units.
3. Sentence Transformers (`all-MiniLM-L6-v2`) creates normalized embeddings.
4. Cosine similarity ranks semantic matches.
5. BM25-style scoring rewards matching words and identifiers.
6. Weighted hybrid scoring combines the two signals.

## Ranking formula

`combined = (semantic_weight × semantic_score + lexical_weight × lexical_score) / (sum of weights)`

Default weights: 0.75 semantic and 0.25 lexical. Cosine similarity is mapped from [-1, 1] to [0, 1]. BM25 scores are normalized to [0, 1]. These are experimental starting weights, not proven optimal settings. Tune them using a labeled evaluation set.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Setup (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. The first model load downloads `all-MiniLM-L6-v2`.

## Usage

1. Select ZIP upload, local folder, or public GitHub repository.
2. Provide the source.
3. Enter a natural-language query.
4. Set the semantic weight; lexical weight is the remainder.
5. Optionally filter by file/folder pattern and code-unit type.
6. Inspect ranked matches and component scores.

A local folder path refers to the machine running Streamlit. GitHub support requires Git installed and supports public repositories over HTTPS.

## Tests

```bash
pytest -q
```

## Evaluation plan

Prepare natural-language queries with known relevant functions. Compare BM25-only, semantic-only, and hybrid search. Report Precision@5, Recall@5 and Mean Reciprocal Rank (MRR). Tune weights on a representative test set rather than a single query.

## Limitations

- Python functions and methods only.
- One repository is searched per request.
- Public GitHub repositories only.
- `all-MiniLM-L6-v2` is a general-purpose sentence embedding model, not a code-specialized retrieval model.
- Similarity scores are ranking signals, not calibrated probabilities.
- Embeddings are rebuilt for each search; persistent and incremental indexing are future improvements.
- Avoid uploading confidential source code to untrusted deployments.
