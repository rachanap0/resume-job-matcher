import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Config
# -----------------------
JOBS_CSV = "data/raw/jobs.csv"
RESUME_PATH = "data/resumes/resume_1.txt"
OUTPUT_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "top_matches_semantic.csv")

TEXT_COLUMN = "Descriptions"
DROP_COLUMNS = ["Unnamed: 0"]

TOP_K = 10
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # strong baseline, fast


def load_jobs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.drop(columns=DROP_COLUMNS, errors="ignore")


def load_resume(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_job_text(df: pd.DataFrame, text_col: str = TEXT_COLUMN) -> pd.Series:
    if text_col not in df.columns:
        raise KeyError(f"Missing '{text_col}' column. Found: {list(df.columns)}")
    return df[text_col].fillna("")


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    df = load_jobs(JOBS_CSV)
    job_text = get_job_text(df)
    resume_text = load_resume(RESUME_PATH)

    model = SentenceTransformer(MODEL_NAME)

    # Encode resume + jobs
    resume_emb = model.encode([resume_text], normalize_embeddings=True)
    job_embs = model.encode(job_text.tolist(), normalize_embeddings=True, show_progress_bar=True)

    scores = cosine_similarity(resume_emb, job_embs).flatten()
    df["match_score"] = scores

    top = df.sort_values("match_score", ascending=False).head(TOP_K).copy()
    top.to_csv(OUTPUT_CSV, index=False)

    cols = ["Title", "Company", "Location", "match_score"]
    existing = [c for c in cols if c in top.columns]
    print(top[existing].to_string(index=False))


if __name__ == "__main__":
    main()
