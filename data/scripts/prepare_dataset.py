import os
import sqlite3
import pandas as pd
import json
from pathlib import Path
# --- Step 1: Load all data into df ---
rows = []
for filename in ['dev.json', 'train_others.json', 'test.json', 'train_spider.json']:
    raw_path = Path("..") / "raw" / filename
    with open(raw_path, 'r') as file:
        data = json.load(file)
        for item in data:
            if isinstance(item, dict) and all(k in item for k in ("db_id", "question", "query")):
                rows.append([item["db_id"], item["question"], item["query"]])

df = pd.DataFrame(rows, columns=["db_id", "question", "query"])

# --- Step 2: Filter valid rows ---
valid_rows = []


for index, row in df.iterrows():
    db_id = row['db_id']
    question = row['question']
    query = row['query']

    db_path= Path("..") / "database" / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        continue  # Skip missing DB files

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            _ = cursor.fetchall()
        # If no exception, add to valid rows
        valid_rows.append(row)
    except sqlite3.Error:
        continue  # Skip rows where query fails

# --- Step 3: Create cleaned DataFrame ---
df_clean = pd.DataFrame(valid_rows)
print(f"Cleaned DataFrame has {len(df_clean)} rows (out of {len(df)})")

#print(df_clean.head(10)) #play a sample of 10 rows from the cleaned DataFrame

save_path = Path("..") / "processed" / "cleaned_dataset.csv"

# Optional: save cleaned DataFrame
df_clean.to_csv(save_path, index=False)
