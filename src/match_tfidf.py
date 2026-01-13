import os
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


def compute_similarity_scores(job_text: pd.Series, resume_text: str) -> pd.Series:
    vectorizer = build_vectorizer()
    job_matrix = vectorizer.fit_transform(job_text)
    resume_vec = vectorizer.transform([resume_text])
    scores = cosine_similarity(resume_vec, job_matrix).flatten()
    return pd.Series(scores, index=job_text.index, name="match_score")


def top_matches(df: pd.DataFrame, resume_text: str, k: int = TOP_K) -> pd.DataFrame:
    job_text = get_job_text(df)
    scores = compute_similarity_scores(job_text, resume_text)

    out = df.copy()
    out["match_score"] = scores

    return out.sort_values("match_score", ascending=False).head(k)


# -----------------------
# Entrypoint
# -----------------------
def main() -> None:
    df = load_jobs(JOBS_CSV)
    resume_text = load_resume(RESUME_PATH)

    matches = top_matches(df, resume_text, k=TOP_K)

    ensure_output_dir(OUTPUT_DIR)
    matches.to_csv(OUTPUT_CSV, index=False)

    cols_to_show = ["Title", "Company", "Location", "match_score"]
    existing_cols = [c for c in cols_to_show if c in matches.columns]
    print(matches[existing_cols].to_string(index=False))


if __name__ == "__main__":
    main()
