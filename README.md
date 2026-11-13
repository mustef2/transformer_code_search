
### ---Group members---

  Name-------------------ID
1. Afomia Birhamu......ATE/6823/14
2. Eyob mulugeta.......ATE/4878/14
3. Mohammed Amino......ATE/4497/14
4. Mustefa Meka........ATE/4355/14



# Code Search Using Transformer Embeddings

A Python-based semantic code search application that helps developers find relevant code by describing what they are looking for in natural language. Instead of relying only on exact keyword matches, the system uses transformer-based embeddings to identify code with similar meaning and functionality.

## 1. Project Overview

**Code Search Using Transformer Embeddings** is a Streamlit application designed to search Python source code across local projects, uploaded ZIP archives, and public GitHub repositories.

The application combines transformer-based semantic search with lexical search using BM25. It supports three search modes:

- **Semantic Search:** Finds code based on meaning and conceptual similarity.
- **Lexical Search:** Finds code based on matching keywords and terms.
- **Hybrid Search:** Combines semantic and lexical relevance scores to produce ranked results.

The project also includes persistent indexing, incremental updates, evaluation metrics, and performance benchmarking.

## 2. Features

- **Natural-language code search:** Describe the functionality you need instead of remembering exact function names.
- **Transformer embeddings:** Uses the `all-MiniLM-L6-v2` sentence-transformer model to represent code as numerical vectors.
- **Python AST parsing:** Extracts functions, classes, and other supported code units from Python source files.
- **Multiple code sources:**
  - Local Python project directories
  - Uploaded ZIP archives
  - Public GitHub repositories
- **Three search modes:** Semantic, lexical, and hybrid.
- **BM25 lexical ranking:** Measures keyword relevance.
- **Hybrid ranking:** Combines semantic and lexical scores.
- **Search filters:** Filter results by file path and symbol type.
- **Result explanations:** Displays relevance scores and explanations for search results.
- **Persistent indexing:** Stores indexed code in SQLite with embeddings stored using NumPy.
- **Incremental indexing:** Uses SHA-256 file hashes to identify changed files and avoid unnecessary reprocessing.
- **Removed-file detection:** Updates the index when previously indexed files are removed.
- **Evaluation:** Measures Precision@K, Recall@K, Reciprocal Rank, and Mean Reciprocal Rank.
- **Performance benchmarking:** Compares the execution time of different search strategies.
- **Automated tests:** Includes tests for parsing, search, database operations, and evaluation.

## 3. Technologies Used

Python -> Main programming language
Streamlit -> Web application interface
Sentence Transformers -> Transformer-based code embeddings
all-MiniLM-L6-v2 -> Embedding model
NumPy -> Vector operations and embedding storage
SQLite -> Persistent index database
Python AST -> Source-code parsing
BM25 -> Lexical search ranking
pytest -> Automated testing
Git and GitHub -> Version control and project hosting

## 4. System Architecture

The application follows a modular architecture:

1. **Source Collection:** Accepts Python projects from local folders, ZIP uploads, and public GitHub repositories.
2. **Code Parsing:** Uses Python AST to extract code units and relevant metadata.
3. **Embedding Generation:** Converts extracted code into vector representations using the transformer model.
4. **Persistent Indexing:** Stores code metadata and embeddings, using file hashes to support incremental updates.
5. **Search Processing:** Executes semantic, lexical, or hybrid search against the indexed code.
6. **Ranking and Filtering:** Ranks matching code units and applies the selected filters.
7. **Result Presentation:** Displays matching code, relevance scores, and explanations in the Streamlit interface.
8. **Evaluation and Benchmarking:** Measures search quality and execution time across different strategies.

## 5. Project Structure

The main project components are organized as follows:

text
transformer_code_search/
│
├── app.py
├── run_evaluation.py
├── run_performance_benchmark.py
├── evaluation_queries.json
│
├── code_search/
│ ├── parser.py
│ ├── search.py
│ ├── database.py
│ ├── github.py
│ ├── sources.py
│ └── evaluation.py
│
├── tests/
│ ├── test_parser.py
│ ├── test_search.py
│ ├── test_database.py
│ └── test_evaluation.py
│
├── performance_results/
│ ├── performance_results.csv
│ └── performance_results.json
│
├── requirements.txt
├── .gitignore
└── README.md

_Note: This is a representation of the main project files. Additional generated files and directories may exist locally._

## 6. Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- Internet connection for installing dependencies and downloading the transformer model

### Step 1: Clone the repository

````bash
git clone https://github.com/mustef2/transformer_code_search.git
cd transformer_code_search


### Step 2: Create a virtual environment

```bash
python -m venv venv


### Step 3: Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
````

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the application

```bash
streamlit run app.py
```

The application will open in your browser. If it does not open automatically, use the local URL displayed in the terminal.

The first run may take longer because the transformer model needs to be downloaded and loaded.

## 7. How to Use the Application

1. Launch the Streamlit application.
2. Select a source for the Python code:
   - Local project directory
   - ZIP archive upload
   - Public GitHub repository
3. Allow the application to parse and index the source code.
4. Enter a natural-language query describing the code functionality you want to find.
5. Select a search mode:
   - Semantic
   - Lexical
   - Hybrid
6. Apply optional file-path and symbol-type filters.

## 8. Search and Ranking Methods

### Semantic Search

Semantic search uses transformer embeddings to compare the meaning of a query with the meaning of indexed code.

The project uses the `all-MiniLM-L6-v2` model to generate embeddings and cosine similarity to measure their similarity.

### Lexical Search

Lexical search uses BM25 scoring to evaluate the relevance of code based on matching terms.

The search implementation also supports tokenization of identifiers, including camelCase and snake_case names.

### Hybrid Search

Hybrid search combines semantic and lexical scores:

\[
S*{\text{hybrid}} =
w_s S*{\text{semantic}} +
w*l S*{\text{lexical}}
\]

Where:

- \(S\_{\text{semantic}}\) is the semantic relevance score.
- \(S\_{\text{lexical}}\) is the lexical relevance score.
- \(w_s\) is the semantic weight.
- \(w_l\) is the lexical weight.

The default application configuration uses a 75% semantic and 25% lexical weighting. Different weighting strategies were evaluated to identify their effect on search quality.

## 9. Evaluation

The project evaluates search quality using manually labeled benchmark queries.

The evaluation includes:

- **Precision@K:** Measures the proportion of retrieved results that are relevant.
- **Recall@K:** Measures the proportion of relevant results successfully retrieved.
- **Reciprocal Rank:** Measures the position of the first relevant result.
- **Mean Reciprocal Rank (MRR):** Measures the average reciprocal rank across queries.

### Evaluation Results

The following results were obtained using 10 manually labeled queries with K = 5:

| Search Strategy | Precision@5 | Recall@5 |    MRR |
| --------------- | ----------: | -------: | -----: |
| Semantic        |      0.3200 |   0.7633 | 1.0000 |
| Lexical         |      0.2400 |   0.6700 | 0.9250 |
| Hybrid 90/10    |      0.3200 |   0.7767 | 1.0000 |
| Hybrid 85/15    |      0.3200 |   0.7767 | 1.0000 |
| Hybrid 80/20    |      0.2800 |   0.7100 | 1.0000 |
| Hybrid 75/25    |      0.2800 |   0.7100 | 1.0000 |
| Hybrid 70/30    |      0.2800 |   0.7100 | 1.0000 |

The 90/10 and 85/15 hybrid configurations achieved the highest Recall@5 while matching the best Precision@5 and MRR in this benchmark.

These results are preliminary and specific to the benchmark dataset and its 10 manually labeled queries. They should not be interpreted as a guarantee of performance across all Python repositories.

## 10. Running Tests

The project uses pytest for automated testing.

Run all tests using:

```bash
python -m pytest -v
```

The latest verified test run completed successfully:

```text
16 passed
```

The test suite covers:

- Python source-code parsing
- Semantic, lexical, and hybrid search
- Search filters and ranking behavior
- Persistent database operations
- Evaluation metric calculations

### ---Group members---

  Name-------------------ID
1. Afomia Birhamu......ATE/6823/14
2. Eyob mulugeta.......ATE/4878/14
3. Mohammed Amino......ATE/4497/14
4. Mustefa Meka........ATE/4355/14
