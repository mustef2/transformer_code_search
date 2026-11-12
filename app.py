
import tempfile
from pathlib import Path

import streamlit as st

from code_search.github import clone_public_github_repo
from code_search.database import PersistentCodeIndex
from code_search.parser import extract_python_functions
from code_search.search import search_functions
from code_search.sources import extract_uploaded_zip


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Transformer Code Search",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 AI-Powered Code Search")

st.caption(
    "Search Python code by meaning using Transformer embeddings "
    "and hybrid ranking."
)


# --------------------------------------------------
# LOAD TRANSFORMER MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# SIDEBAR: CODE SOURCE AND SEARCH CONFIGURATION
# --------------------------------------------------

with st.sidebar:

    st.header("Code source")

    source_type = st.radio(
        "Repository source",
        [
            "Upload ZIP",
            "Local folder",
            "Public GitHub repository"
        ]
    )

    uploaded_zip = None
    local_path = ""
    github_url = ""

    if source_type == "Upload ZIP":

        uploaded_zip = st.file_uploader(
            "Upload a ZIP containing Python code",
            type=["zip"]
        )

    elif source_type == "Local folder":

        local_path = st.text_input(
            "Folder path on the machine running Streamlit"
        )

    else:

        github_url = st.text_input(
            "Public GitHub URL",
            placeholder="https://github.com/owner/repository"
        )

    st.divider()

    # ----------------------------------------------
    # RANKING CONFIGURATION
    # ----------------------------------------------

    st.header("Ranking")

    search_mode = st.selectbox(
        "Search strategy",
        ["hybrid", "semantic", "lexical"],
        format_func=lambda mode: {
            "hybrid": "Hybrid (Semantic + Lexical)",
            "semantic": "Semantic only",
            "lexical": "Lexical only"
        }[mode],
        index=0
    )

    if search_mode == "hybrid":

        semantic_weight = st.slider(
            "Semantic weight",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05
        )

        lexical_weight = 1.0 - semantic_weight

        st.caption(
            f"Lexical weight: {lexical_weight:.2f}"
        )

    else:

        semantic_weight = 0.75
        lexical_weight = 0.25

        if search_mode == "semantic":
            st.caption(
                "Ranking uses Transformer embeddings only."
            )
        else:
            st.caption(
                "Ranking uses BM25 keyword matching only."
            )

    result_count = st.slider(
        "Number of results",
        min_value=1,
        max_value=20,
        value=5
    )

    st.divider()

    # ----------------------------------------------
    # SEARCH FILTERS
    # ----------------------------------------------

    st.header("Filters")

    file_filter = st.text_input(
        "File/folder pattern",
        placeholder="services/*.py or api/",
        help="Optional relative path filter. Leave blank to search all files."
    )

    kind = st.selectbox(
        "Code unit type",
        [
            "All",
            "Functions",
            "Methods",
            "Async functions",
            "Async methods"
        ]
    )


# --------------------------------------------------
# REPOSITORY SOURCE HANDLING
# --------------------------------------------------

def get_repository_path():

    if source_type == "Upload ZIP":

        if uploaded_zip is None:

            st.warning("Upload a ZIP file first.")

            return None

        temp_root = Path(
            tempfile.mkdtemp(prefix="code_search_")
        )

        archive = temp_root / uploaded_zip.name

        archive.write_bytes(
            uploaded_zip.getvalue()
        )

        return extract_uploaded_zip(
            archive,
            temp_root / "extracted"
        )

    elif source_type == "Local folder":

        path = Path(local_path).expanduser()

        if not local_path.strip() or not path.is_dir():

            st.error(
                "Enter an existing folder accessible to the machine running Streamlit."
            )

            return None

        return path.resolve()

    else:

        if not github_url.strip():

            st.warning(
                "Enter a public GitHub repository URL."
            )

            return None

        with st.spinner(
            "Cloning public GitHub repository..."
        ):

            return clone_public_github_repo(
                github_url.strip()
            )


# --------------------------------------------------
# PERSISTENT CODE INDEX
# --------------------------------------------------

st.header("Persistent index")

st.caption(
    "SQLite stores extracted code and embeddings locally "
    "so unchanged files are not re-encoded."
)

database_path = st.text_input(
    "SQLite database path",
    value="data/code_search.db"
)


# --------------------------------------------------
# SEARCH INTERFACE
# --------------------------------------------------

st.header("Search your code")

query = st.text_input(
    "Describe what the code should do",
    placeholder="Find a function that validates an email address"
)


# --------------------------------------------------
# SEARCH EXECUTION
# --------------------------------------------------

if st.button(
    "Search code",
    type="primary",
    disabled=not bool(query.strip())
):

    try:

        # ------------------------------------------
        # GET REPOSITORY
        # ------------------------------------------

        repo_path = get_repository_path()

        if repo_path is not None:

            # --------------------------------------
            # LOAD MODEL AND UPDATE INDEX
            # --------------------------------------

            with st.spinner(
                "Loading Transformer and updating the persistent index..."
            ):

                model = load_model()

                index = PersistentCodeIndex(
                    database_path
                )

                documents, embeddings, stats = (
                    index.index_repository(
                        repo_path,
                        model
                    )
                )

            st.success(
                f"Indexed {stats['indexed_code_units']} code units "
                f"across {stats['total_files']} Python files. "
                f"Changed: {stats['changed_files']} · "
                f"Reused: {stats['reused_files']} · "
                f"Removed: {stats['removed_files']}."
            )

            # --------------------------------------
            # CHECK FOR CODE UNITS
            # --------------------------------------

            if not documents:

                st.warning(
                    "No Python functions were found."
                )

            else:

                # ----------------------------------
                # CODE UNIT FILTER MAPPING
                # ----------------------------------

                kind_map = {

                    "All": None,

                    "Functions": "function",

                    "Methods": "method",

                    "Async functions": "async_function",

                    "Async methods": "async_method"

                }

                # ----------------------------------
                # SEARCH AND RANK RESULTS
                # ----------------------------------

                with st.spinner(
                    "Ranking relevant code..."
                ):

                    results = search_functions(

                        query,

                        documents,

                        embeddings,

                        model,

                        top_k=result_count,

                        semantic_weight=semantic_weight,

                        lexical_weight=lexical_weight,

                        file_filter=file_filter.strip() or None,

                        symbol_type=kind_map[kind],

                        search_mode=search_mode

                    )

                # ----------------------------------
                # DISPLAY SEARCH STRATEGY
                # ----------------------------------

                if search_mode == "hybrid":

                    st.caption(
                        f"Hybrid ranking: "
                        f"{semantic_weight:.0%} semantic + "
                        f"{lexical_weight:.0%} lexical. "
                        f"Scores are not probabilities."
                    )

                elif search_mode == "semantic":

                    st.caption(
                        "Semantic-only ranking using Transformer embeddings. "
                        "Scores are not probabilities."
                    )

                else:

                    st.caption(
                        "Lexical-only ranking using BM25 keyword matching. "
                        "Scores are not probabilities."
                    )

                # ----------------------------------
                # DISPLAY RESULTS
                # ----------------------------------

                if not results:

                    st.warning(
                        "No code units matched the selected filters."
                    )

                else:

                    st.subheader("Best matches")

                    st.write(
                        f"Found {len(results)} relevant code results."
                    )

                    for rank, result in enumerate(
                        results,
                        start=1
                    ):

                        doc = result["document"]

                        with st.expander(

                            f"{rank}. {doc['qualified_name']} "
                            f"· score {result['score']:.3f}",

                            expanded=(rank == 1)

                        ):

                            # ----------------------
                            # FILE INFORMATION
                            # ----------------------

                            st.caption(

                                f"{doc['file_path']} · "
                                f"lines {doc['start_line']}–"
                                f"{doc['end_line']} · "
                                f"{doc['symbol_type']}"

                            )

                            # ----------------------
                            # SEARCH SCORES
                            # ----------------------

                            a, b, c = st.columns(3)

                            a.metric(
                                "Combined",
                                f"{result['score']:.3f}"
                            )

                            b.metric(
                                "Semantic",
                                f"{result['semantic_score']:.3f}"
                            )

                            c.metric(
                                "Lexical",
                                f"{result['lexical_score']:.3f}"
                            )

                            # ----------------------
                            # EXPLANATION
                            # ----------------------

                            st.write(
                                "**Why this result?**",
                                result["explanation"]
                            )

                            # ----------------------
                            # DOCSTRING
                            # ----------------------

                            if doc["docstring"]:

                                st.write(
                                    "**Docstring:**",
                                    doc["docstring"]
                                )

                            # ----------------------
                            # SOURCE CODE
                            # ----------------------

                            st.write("**Source code:**")

                            st.code(
                                doc["source"],
                                language="python"
                            )

    except Exception as exc:

        st.error(
            f"Could not search this source: {exc}"
        )

        st.info(
            "Check the ZIP/folder or confirm the GitHub URL "
            "points to a public repository."
        )

else:

    st.info(
        "Choose a source, enter a natural-language query, "
        "and select Search code."
    )


# --------------------------------------------------
# HOW SEARCH WORKS
# --------------------------------------------------

with st.expander("How hybrid search works"):

    st.write(

        "The Transformer encoder creates dense vectors for "
        "the query and each code unit. "

        "Cosine similarity captures semantic meaning. "

        "BM25-style lexical scoring rewards matching terms "
        "and identifiers. "

        "A weighted combination ranks the results."

    )

    st.write("**Available search strategies:**")

    st.markdown(
        """
        - **Semantic:** Finds code based on meaning and context.
        - **Lexical:** Finds code based on matching keywords and identifiers.
        - **Hybrid:** Combines semantic and lexical scores for ranking.
        """
    )
