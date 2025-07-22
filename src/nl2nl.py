import os
import numpy as np
import time
from vllm import LLM, SamplingParams
import pandas as pd
from transformers import AutoTokenizer
from dotenv import load_dotenv
from pathlib import Path
from sql_utils import generate_simplified_schema

load_dotenv()  # Load environment variables from .env file

# Set threading layer early
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ['HF_TOKEN'] = os.getenv("API_KEY")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("API_KEY")


# === Change model and tokenizer here ===
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKENIZER_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# Global variables (lazy-loaded)
_llm_nl2nl = None  # Singleton LLM instance
_sampling_nl2nl = SamplingParams(temperature=0, max_tokens=100)  # Sampling params for LLM
_tokenizer_nl2nl = None  # Singleton tokenizer instance

project_root = Path(__file__).resolve().parent.parent

def get_tokenizer():
    """Return a singleton tokenizer instance for paraphrasing."""
    global _tokenizer_nl2nl
    if _tokenizer_nl2nl is None:
        _tokenizer_nl2nl = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME,
            trust_remote_code=True
        )
    return _tokenizer_nl2nl

def get_llm_nl2nl():
    """Return a singleton LLM instance for paraphrasing."""
    global _llm_nl2nl
    if _llm_nl2nl is None:
        _llm_nl2nl = LLM(
            model=MODEL_NAME,
            max_model_len=2048,
            tokenizer=TOKENIZER_NAME,
            hf_token=os.environ['HF_TOKEN'],
            trust_remote_code=True,
            tensor_parallel_size=2  # Parallel
        )
    return _llm_nl2nl


def paraphrase_sentence(sentence: str, schema: str = "") -> str:
    """
    Paraphrase an NL question while guaranteeing SQL-equivalent semantics.
    Falls back to the original if any semantic-drift guard fails.
    """
    tokenizer = get_tokenizer()

    sys_msg = (
        "You are a helpful assistant for paraphrasing natural language questions that will be converted into SQL queries. Rewrite the question using different words or phrasing, but make sure the meaning and logic are exactly the same so that both the original and paraphrased versions would generate the same SQL query.\n\n"
        "- Keep all columns, tables, filters, and conditions unchanged.\n"
        "- Do not add, remove, or change any information.\n"
        "- Do not change quantifiers (like 'each', 'every', 'distinct', 'any').\n"
        "- Do not change time or comparison logic (e.g., 'after 2013' must stay 'after 2013').\n"
        "- Do not expand or shorten abbreviations keep it unchanged.\n\n"
        "- Do NOT change the casing of string values. For example, keep 'france' as 'france' — do not convert to 'FRANCE' or 'France'."
        "Return only the paraphrased question as plain text, with no explanation or extra text."
    )

    usr_msg = (
        f"Here is the table schema for context:\n{schema}\n\n"
        f"Original question:\n{sentence}"
    )

    chat_input = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_msg},
         {"role": "user", "content": usr_msg}],
        tokenize=False,
        add_generation_prompt=True
    )

    llm = get_llm_nl2nl()
    try:
        outputs = llm.generate(chat_input, sampling_params=_sampling_nl2nl)
        para = outputs[0].outputs[0].text.strip()
        return para
    except Exception as e:
        print(f"[paraphrase_sentence] Error: {e}")
        return sentence


if __name__ == "__main__":
    from multiprocessing import set_start_method
    set_start_method("spawn", force=True)

    start_time = time.time()
    # Load input data
    df_path= project_root/ "data" / "processed" / "final_dataset.csv"
    df = pd.read_csv(df_path)


    new_df = df.copy()
    for i, row in new_df.iterrows():
        question = row['question']
        schema = generate_simplified_schema(row['db_id'])
        try:
            paraphrased = paraphrase_sentence(question, schema)
        except Exception as e:
            print(f"[Row {i}] Paraphrasing error: {e}")
            paraphrased = question
        new_df.loc[i, "paraphrased_nl"] = paraphrased
        print(f"Original: {question}")
        print(f"Paraphrased: {paraphrased}")

    end_time = time.time()
    print(f"Total time taken for paraphrasing: {end_time - start_time} seconds")
    
    # Save results to CSV
    new_path= project_root/ "data" /"processed"/"output_paraphrased.csv"
    new_df.to_csv(new_path, index=False)