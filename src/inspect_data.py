import pandas as pd

CSV_PATH = "data/raw/jobs.csv"

def main():
    #load the CSV file into a DataFrame
    df = pd.read_csv(CSV_PATH)

    print("Number of rows", len(df))
    print("Columns:" )
    for col in df.columns:
        print("-", col)

    print("\nSample rows:")
    print(df.head(3))

if __name__ == "__main__":
    main()