import os
import time
import pandas as pd

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sql_utils import run_all, generate_simplified_schema 
from dotenv import load_dotenv
from pathlib import Path

load_dotenv() 

# Set environment
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ['HF_TOKEN'] = os.getenv("API_KEY")


# === Change model and tokenizer here ===
MODEL_NAME = "XGenerationLab/XiYanSQL-QwenCoder-7B-2504"  # Model name for LLM
TOKENIZER_NAME = "XGenerationLab/XiYanSQL-QwenCoder-7B-2504"  # Tokenizer name for LLM



# Global variables (lazy-loaded)
_llm_nl2sql = None
_tokenizer = None
_sampling_sql = SamplingParams(temperature=0, max_tokens=256)

project_root = Path(__file__).resolve().parent.parent

def get_llm_nl2sql():
    """Return a singleton LLM instance for SQL generation."""
    global _llm_nl2sql
    if _llm_nl2sql is None:
        _llm_nl2sql = LLM(
            model=MODEL_NAME,
            max_model_len=1024,
            tokenizer=TOKENIZER_NAME,
            #gpu_memory_utilization=0.85,  # Uncomment if needed
            trust_remote_code=True,
            tensor_parallel_size=2,
        )
    return _llm_nl2sql

def get_tokenizer():
    """Return a singleton tokenizer instance for SQL generation."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME,
            trust_remote_code=True
        )
    return _tokenizer

def generate_sql(db_name: str,question: str) -> str:
    """Generate SQL from a natural language prompt using the LLM."""
    tokenizer = get_tokenizer()
    schema= generate_simplified_schema(db_name)
    chat_input = tokenizer.apply_chat_template([
        {
            "role": "system",
            "content": (
                    f"You are a helpful assistant that generates SQL queries from natural language questions and a database schema.\n\n"
                    "RULES:\n"
                    "- Use only tables and columns from the schema.\n"
                    "- Do NOT hallucinate or infer columns, joins, or filters.\n"
                    "- Use explicit JOINs when needed.\n"
                    "- Return ONLY the SQL query (no explanation).\n\n"
                    f"Here is the schema:\n{schema}\n"
                "Example:\n"
                "-- User question: 'Get the model name and price of all cars.'\n"
                "-- Schema tables: CAR_NAMES (model_id, model), CAR_PRICES (model_id, price)\n"
                "Output:\n"
                "SELECT CAR_NAMES.model, CAR_PRICES.price FROM CAR_NAMES JOIN CAR_PRICES ON CAR_NAMES.model_id = CAR_PRICES.model_id;"
            )
        },
        {"role": "user", "content": f"Here is the question {question}."}
    ], tokenize=False, add_generation_prompt=True)
    llm = get_llm_nl2sql()
    try:
        outputs = llm.generate([chat_input], sampling_params=_sampling_sql)
        return outputs[0].outputs[0].text.strip()
    except Exception as e:
        print(f"[generate_sql] Error: {e}")
        return ""

def reverifier(initial_sql: str, db_id: str, gt_sql: str,question: str, max_retries: int = 5) -> tuple[str, bool, int]:
    """
    Try to correct the SQL up to max_retries times using the LLM.
    Returns (final_sql, matched, attempts). Returns attempts = -1 if failed.
    """
    incorrect_sql = [initial_sql]
    sql = initial_sql
    matched = run_all(db_id, gt_sql, sql)
    attempts = 1

    schema= generate_simplified_schema(db_id)
    tokenizer = get_tokenizer()

    while not matched and attempts <= max_retries:
        chat_input = tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": (
                    "You are a SQL fixer. You are given a natural language question and a list of faulty SQL queries.\n"
                    "Your task is to fix the query without adding assumptions or extra details. Fix only the actual mistakes.\n"
                    "Most wrong queries are caused by unnecessary joins, columns, aggregations, or logic errors.\n"
                    "Fix errors such as invalid columns, join mismatches, or missing clauses. Your fix should be minimal and safe for execution.\n"
                    "Strictly follow the schema. Do not invent columns, tables, or joins not present in the schema.\n"
                    "Make sure the query follows the schema\n."
                    "Return ONLY the fixed SQL query without any explanation.\n\n"   
                )
            },
            {
                "role": "user",
                "content": (
                    f"Here is the Question: {question}.\n\n"
                    f"Faulty SQL queries list (oldest to newest):\n{incorrect_sql}.\n\n"
                    f"Here is the database schema for context:\n{schema}.\n\n"
                )
            }
        ], tokenize=False, add_generation_prompt=True)

        llm = get_llm_nl2sql()

        #print("Reverifier attempt:", attempts)  #Debugging
        #print("Chat input:", chat_input)
        try:
            outputs = llm.generate(chat_input, sampling_params=_sampling_sql)
            sql_new = outputs[0].outputs[0].text.strip()
        except Exception as e:
            print(f"[reverifier] Error: {e}")
            break

        if sql_new == sql:
            break

        sql = sql_new
        incorrect_sql.append(sql)
        matched = run_all(db_id, gt_sql, sql)
        attempts += 1

    return sql, matched, attempts if matched else -1


# Safe multiprocessing
if __name__ == "__main__":
    from multiprocessing import set_start_method
    set_start_method("spawn", force=True)

    start = time.time()

    # Load input data
    df_path= project_root / "data" / "processed" / "output_paraphrased.csv"
    df = pd.read_csv(df_path)

    new_df = df.copy()

    for i, row in new_df.iterrows():
        # Extract fields for this row
        question = row.get('paraphrased_nl', '') 
        gt_sql = row.get('query', '')
        db_id = row.get('db_id', '')

        # Generate SQL
        try:
            generated_sql = generate_sql(db_id,question)
        except Exception as e:
            print(f"[{i}] Generation error: {e}")
            generated_sql = ""

        attempts = 1
        try:
            matched = run_all(db_id, gt_sql, generated_sql)
            if matched:
                attempts = 1
            else:
                #Reverify the generated SQL if it doesn't match
                generated_sql, matched, reverifier_attempts = reverifier(generated_sql, db_id, gt_sql,question=question)   # Debugging reverifier
                attempts = reverifier_attempts if reverifier_attempts != -1 else -1
                print("❌ Not Matched")
                print("db_id: ",db_id)
                print("Paraphrased:", question)
                print("GT SQL:", gt_sql)
                print("Generated SQL:", generated_sql)
        except Exception as e:
            print(f"[{i}] Match error: {e}")
            matched = False
            attempts = -1

        # Save results for this row
        new_df.loc[i, "original_nl"] = row.get('question', '')
        new_df.loc[i, "generated_sql"] = generated_sql
        new_df.loc[i, "matched"] = matched
        new_df.loc[i, "attempts"] = attempts
        new_df.loc[i, "gt_sql"] = gt_sql

    end = time.time()
    print(f"Total time taken for SQL generation: {end - start:.2f} seconds")

    # Save all results to CSV (including db_id)
    new_path=project_root / "result" / "results.csv"
    new_df[["db_id", "original_nl", "paraphrased_nl","gt_sql","generated_sql", "matched", "attempts"]].to_csv(new_path, index=False)
