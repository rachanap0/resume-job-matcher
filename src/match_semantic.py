# src/match_semantic.py
import os
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TEXT_COLUMN_DEFAULT = "Descriptions"
DROP_COLUMNS_DEFAULT = ["Unnamed: 0"]
MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"


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
        raise KeyError(f"Missing '{text_col}' column. Found: {list(df.columns)}")
    return df[text_col].fillna("")


def run_semantic_match(
    jobs_csv: str,
    resume_path: str,
    out_csv: str,
    top_k: int = 10,
    text_column: str = TEXT_COLUMN_DEFAULT,
    model_name: str = MODEL_DEFAULT,
    drop_columns=None,
) -> pd.DataFrame:
    drop_columns = drop_columns or DROP_COLUMNS_DEFAULT

    df = load_jobs(jobs_csv, drop_columns=drop_columns)
    job_text = get_job_text(df, text_column)
    resume_text = load_resume(resume_path)

    model = SentenceTransformer(model_name)

    resume_emb = model.encode([resume_text], normalize_embeddings=True)
    job_embs = model.encode(job_text.tolist(), normalize_embeddings=True, show_progress_bar=True)

    scores = cosine_similarity(resume_emb, job_embs).flatten()
    df["match_score"] = scores

    top = df.sort_values("match_score", ascending=False).head(top_k).copy()

    ensure_output_dir(os.path.dirname(out_csv) or ".")
    top.to_csv(out_csv, index=False)

    return top


def main():
    p = argparse.ArgumentParser(description="Semantic resume-job matcher")
    p.add_argument("--jobs", default="data/raw/jobs.csv")
    p.add_argument("--resume", default="data/resumes/resume_1.txt")
    p.add_argument("--out", default="outputs/top_matches_semantic.csv")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--text-col", default=TEXT_COLUMN_DEFAULT)
    p.add_argument("--model", default=MODEL_DEFAULT)
    args = p.parse_args()

    top = run_semantic_match(
        jobs_csv=args.jobs,
        resume_path=args.resume,
        out_csv=args.out,
        top_k=args.top_k,
        text_column=args.text_col,
        model_name=args.model,
    )

    cols = ["Title", "Company", "Location", "match_score"]
    existing = [c for c in cols if c in top.columns]
    print(top[existing].to_string(index=False))


if __name__ == "__main__":
    main()
