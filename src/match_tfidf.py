# src/match_tfidf.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TEXT_COLUMN_DEFAULT = "Descriptions"
DROP_COLUMNS_DEFAULT = ["Unnamed: 0"]


def load_jobs(csv_path: str, drop_columns=None) -> pd.DataFrame:
    drop_columns = drop_columns or []
    df = pd.read_csv(csv_path)
    return df.drop(columns=drop_columns, errors="ignore")


def load_resume(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_job_text(df: pd.DataFrame, text_col: str) -> pd.Series:
    if text_col not in df.columns:
        raise KeyError(
            f"Expected text column '{text_col}' not found. Available columns: {list(df.columns)}"
        )
    return df[text_col].fillna("")


def build_vectorizer(max_features: int = 50_000) -> TfidfVectorizer:
    return TfidfVectorizer(stop_words="english", max_features=max_features)


def top_terms(vectorizer: TfidfVectorizer, vec, n: int = 15) -> list[str]:
    feature_names = vectorizer.get_feature_names_out()
    weights = vec.toarray().ravel()
    if weights.size == 0:
        return []
    top_idx = np.argsort(weights)[::-1][:n]
    return [feature_names[i] for i in top_idx if weights[i] > 0]


def run_tfidf_match(
    jobs_csv: str,
    resume_path: str,
    out_csv: str,
    top_k: int = 10,
    text_column: str = TEXT_COLUMN_DEFAULT,
    max_features: int = 50_000,
    resume_terms_n: int = 30,
    job_terms_n: int = 30,
    overlap_n: int = 12,
    drop_columns=None,
) -> pd.DataFrame:
    drop_columns = drop_columns or DROP_COLUMNS_DEFAULT

    df = load_jobs(jobs_csv, drop_columns=drop_columns)
    resume_text = load_resume(resume_path)
    job_text = get_job_text(df, text_column)

    vectorizer = build_vectorizer(max_features=max_features)
    job_matrix = vectorizer.fit_transform(job_text)
    resume_vec = vectorizer.transform([resume_text])

    scores = cosine_similarity(resume_vec, job_matrix).flatten()
    out = df.copy()
    out["match_score"] = scores

    top_df = out.sort_values("match_score", ascending=False).head(top_k).copy()
    top_indices = top_df.index.tolist()

    resume_terms = set(top_terms(vectorizer, resume_vec, n=resume_terms_n))
    index_to_pos = {idx: pos for pos, idx in enumerate(job_text.index)}

    matched_keywords = []
    missing_keywords = []

    for idx in top_indices:
        pos = index_to_pos.get(idx)
        if pos is None:
            matched_keywords.append("")
            missing_keywords.append("")
            continue

        job_vec = job_matrix[pos]
        job_terms = set(top_terms(vectorizer, job_vec, n=job_terms_n))

        overlap = sorted(resume_terms.intersection(job_terms))[:overlap_n]
        missing = sorted(job_terms - resume_terms)[:overlap_n]

        matched_keywords.append(", ".join(overlap))
        missing_keywords.append(", ".join(missing))

    top_df["matched_keywords"] = matched_keywords
    top_df["missing_keywords"] = missing_keywords

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


def main():
    p = argparse.ArgumentParser(description="TF-IDF resume-job matcher")
    p.add_argument("--jobs", default="data/raw/jobs.csv")
    p.add_argument("--resume", default="data/resumes/resume_1.txt")
    p.add_argument("--out", default="outputs/top_matches_tfidf.csv")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--text-col", default=TEXT_COLUMN_DEFAULT)
    p.add_argument("--max-features", type=int, default=50_000)
    args = p.parse_args()

    top = run_tfidf_match(
        jobs_csv=args.jobs,
        resume_path=args.resume,
        out_csv=args.out,
        top_k=args.top_k,
        text_column=args.text_col,
        max_features=args.max_features,
    )

    cols = ["Title", "Company", "Location", "match_score", "matched_keywords", "missing_keywords"]
    existing = [c for c in cols if c in top.columns]
    print(top[existing].to_string(index=False))


if __name__ == "__main__":
    main()
