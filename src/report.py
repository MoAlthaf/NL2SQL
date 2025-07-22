import pandas as pd
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
RESULTS_PATH = project_root/"result/results.csv"

def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"[ERROR] Results file not found: {RESULTS_PATH}")
        return

    try:
        df = pd.read_csv(RESULTS_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to read results file: {e}")
        return

    if df.empty:
        print("[WARNING] Results file is empty.")
        return

    total = len(df)
    correct = df["matched"].sum() if "matched" in df.columns else 0
    accuracy = (correct / total * 100) if total > 0 else 0.0

    print(f"Total Samples Evaluated: {total}")
    print(f"Correct SQL Matches:     {correct}")
    print(f"Accuracy:                {accuracy:.2f}%")

    # Attempts statistics
    if "attempts" in df.columns:
        attempts_valid = df["attempts"].replace(-1, pd.NA).dropna()
        if not attempts_valid.empty:
            avg_attempts = attempts_valid.mean()
            min_attempts = attempts_valid.min()
            max_attempts = attempts_valid.max()
            print(f"Average Attempts:         {avg_attempts:.2f}")
            print(f"Min Attempts:             {min_attempts}")
            print(f"Max Attempts:             {max_attempts}")
        else:
            print("No valid attempts data available.")
    else:
        print("No 'attempts' column in results.")

if __name__ == "__main__":
    main()
