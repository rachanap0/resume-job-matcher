# src/cli.py
import argparse

from match_semantic import run_semantic_match
from match_tfidf import run_tfidf_match


def main():
    p = argparse.ArgumentParser(prog="resume-job-matcher")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("semantic", help="Run semantic matcher")
    s.add_argument("--jobs", default="data/raw/jobs.csv")
    s.add_argument("--resume", default="data/resumes/resume_1.txt")
    s.add_argument("--out", default="outputs/top_matches_semantic.csv")
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--text-col", default="Descriptions")
    s.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")

    t = sub.add_parser("tfidf", help="Run TF-IDF matcher")
    t.add_argument("--jobs", default="data/raw/jobs.csv")
    t.add_argument("--resume", default="data/resumes/resume_1.txt")
    t.add_argument("--out", default="outputs/top_matches_tfidf.csv")
    t.add_argument("--top-k", type=int, default=10)
    t.add_argument("--text-col", default="Descriptions")
    t.add_argument("--max-features", type=int, default=50_000)

    args = p.parse_args()

    if args.cmd == "semantic":
        run_semantic_match(
            jobs_csv=args.jobs,
            resume_path=args.resume,
            out_csv=args.out,
            top_k=args.top_k,
            text_column=args.text_col,
            model_name=args.model,
        )
        print(f"Saved: {args.out}")

    elif args.cmd == "tfidf":
        run_tfidf_match(
            jobs_csv=args.jobs,
            resume_path=args.resume,
            out_csv=args.out,
            top_k=args.top_k,
            text_column=args.text_col,
            max_features=args.max_features,
        )
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
