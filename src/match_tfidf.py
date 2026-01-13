import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Config (easy to tweak)
# -----------------------
JOBS_CSV = "data/raw/jobs.csv"
RESUME_PATH = "data/resumes/resume_1.txt"
OUTPUT_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "top_matches_tfidf.csv")

TEXT_COLUMN = "Descriptions"
DROP_COLUMNS = ["Unnamed: 0"]

TOP_K = 10
MAX_FEATURES = 50_000

# Explanation knobs
RESUME_TERMS_N = 30   # how many top resume terms to consider
JOB_TERMS_N = 30      # how many top job terms to consider
OVERLAP_N = 12        # how many overlapping terms to display


# -----------------------
# I/O helpers
# -----------------------
def load_jobs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.drop(columns=DROP_COLUMNS, errors="ignore")


def load_resume(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------
# Core matching logic
# -----------------------
def get_job_text(df: pd.DataFrame, text_col: str = TEXT_COLUMN) -> pd.Series:
    if text_col not in df.columns:
        raise KeyError(
            f"Expected text column '{text_col}' not found. Available columns: {list(df.columns)}"
        )
    return df[text_col].fillna("")


def build_vectorizer(max_features: int = MAX_FEATURES) -> TfidfVectorizer:
    return TfidfVectorizer(stop_words="english", max_features=max_features)


def top_terms(vectorizer: TfidfVectorizer, vec, n: int = 15) -> list[str]:
    """
    Return top-n terms (by TF-IDF weight) from a single sparse vector.
    """
    feature_names = vectorizer.get_feature_names_out()
    weights = vec.toarray().ravel()
    if weights.size == 0:
        return []
    top_idx = np.argsort(weights)[::-1][:n]
    return [feature_names[i] for i in top_idx if weights[i] > 0]


def explain_matches(df: pd.DataFrame, resume_text: str, k: int = TOP_K) -> pd.DataFrame:
    """
    Computes TF-IDF cosine similarity and adds an explanation column:
    'matched_keywords' = overlap of top TF-IDF terms (resume vs job).
    """
    job_text = get_job_text(df)

    vectorizer = build_vectorizer()
    job_matrix = vectorizer.fit_transform(job_text)
    resume_vec = vectorizer.transform([resume_text])

    scores = cosine_similarity(resume_vec, job_matrix).flatten()

    out = df.copy()
    out["match_score"] = scores

    top_df = out.sort_values("match_score", ascending=False).head(k).copy()
    top_indices = top_df.index.tolist()

    # Terms for resume (once)
    resume_terms = set(top_terms(vectorizer, resume_vec, n=RESUME_TERMS_N))

    matched_keywords = []
    missing_keywords = []

    # Map dataframe index -> row position inside job_text/job_matrix
    index_to_pos = {idx: pos for pos, idx in enumerate(job_text.index)}

    for idx in top_indices:
        pos = index_to_pos.get(idx)
        if pos is None:
            matched_keywords.append("")
            missing_keywords.append("")
            continue

        job_vec = job_matrix[pos]
        job_terms = set(top_terms(vectorizer, job_vec, n=JOB_TERMS_N))

        overlap = sorted(resume_terms.intersection(job_terms))[:OVERLAP_N]
        missing = sorted(job_terms - resume_terms)[:OVERLAP_N]

        matched_keywords.append(", ".join(overlap))
        missing_keywords.append(", ".join(missing))

    top_df["matched_keywords"] = matched_keywords
    top_df["missing_keywords"] = missing_keywords
    return top_df

def run_tfidf_match(
    jobs_csv: str,
    resume_path: str,
    out_csv: str,
    top_k: int = 10,
    text_column: str = "Descriptions",
    max_features: int = 50_000,
) -> pd.DataFrame:
    # update globals used by helpers (keeps changes minimal)
    global TEXT_COLUMN, MAX_FEATURES
    TEXT_COLUMN = text_column
    MAX_FEATURES = max_features

    df = load_jobs(jobs_csv)
    resume_text = load_resume(resume_path)

    matches = explain_matches(df, resume_text, k=top_k)

    ensure_output_dir(os.path.dirname(out_csv) or ".")
    matches.to_csv(out_csv, index=False)
    return matches
# -----------------------
# Entrypoint
# -----------------------
def main() -> None:
    df = load_jobs(JOBS_CSV)
    resume_text = load_resume(RESUME_PATH)

    matches = explain_matches(df, resume_text, k=TOP_K)

    ensure_output_dir(OUTPUT_DIR)
    matches.to_csv(OUTPUT_CSV, index=False)

    cols_to_show = ["Title", "Company", "Location", "match_score", "matched_keywords", "missing_keywords"]
    existing_cols = [c for c in cols_to_show if c in matches.columns]
    print(matches[existing_cols].to_string(index=False))


if __name__ == "__main__":
    main()
