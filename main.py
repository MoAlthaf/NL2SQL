import subprocess
import time
import pandas as pd
import torch
if __name__ == "__main__":
    start = time.time()
    
    #Setting up dataset
    df=pd.read_csv("dataset/dataset.csv")
    new_df=df.copy()      # Adjust the number of rows as needed
    new_df.to_csv("dataset/final_dataset.csv", index=False)

    print("Running NL2NL...")
    subprocess.run(["python3", "nl2nl.py"], check=True)

    print("Clearing cache...")
    torch.cuda.empty_cache()

    print("Running nl2sql...")
    subprocess.run(["python3", "nl2sql.py"], check=True)

    print(f"Finished everything in {(time.time() - start)/60:.2f} minutes")
