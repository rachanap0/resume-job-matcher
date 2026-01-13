import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

JOBS_CSV = "data/raw/jobs.csv"
RESUME_PATH = "data/resumes/resume_1.txt"


def load_resume(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_job_corpus(df: pd.DataFrame) -> pd.Series:
    return df["Descriptions"].fillna("")


def top_matches_tfidf(df: pd.DataFrame, resume_text: str, k: int = 10) -> pd.DataFrame:
    job_docs = build_job_corpus(df)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    job_matrix = vectorizer.fit_transform(job_docs)
    resume_vec = vectorizer.transform([resume_text])

    scores = cosine_similarity(resume_vec, job_matrix).flatten()
    top_idx = scores.argsort()[::-1][:k]

    results = df.iloc[top_idx].copy()
    results["match_score"] = scores[top_idx]
    return results.sort_values("match_score", ascending=False)


def main():
    df = pd.read_csv(JOBS_CSV)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    resume_text = load_resume(RESUME_PATH)
    matches = top_matches_tfidf(df, resume_text, k=10)

    # Save results
    matches.to_csv("outputs/top_matches_tfidf.csv", index=False)

    # Print results
    print(
    matches[["Title", "Company", "Location", "match_score"]]
    .to_string(index=False)
)



if __name__ == "__main__":
    main()
