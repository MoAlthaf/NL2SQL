import subprocess
import time
import pandas as pd
import torch
from pathlib import Path
import os

if __name__ == "__main__":
    start = time.time()
    
    project_root = Path(__file__).resolve().parent.parent
    #Setting up dataset
    df_path=project_root/ "data" / "processed" / "cleaned_dataset.csv"
    df=pd.read_csv(df_path)
    new_df=df.copy()     # Adjust the number of rows as needed

    new_df_path=project_root/ "data" / "processed" / "final_dataset.csv"
    new_df.to_csv(new_df_path, index=False)

    print("Running NL2NL...")
    subprocess.run(["python3", f"{project_root}/src/nl2nl.py"], check=True)

    print("Clearing cache...")
    torch.cuda.empty_cache()

    print("Running nl2sql...")
    subprocess.run(["python3", "src/nl2sql.py"], check=True)

    print(f"Finished everything in {(time.time() - start)/60:.2f} minutes")
