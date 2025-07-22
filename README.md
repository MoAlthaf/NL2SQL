# NL2SQL: Natural Language to SQL via LLMs

## Task Description

This project implements a two-stage pipeline using large language models (LLMs) to translate natural language questions into executable SQL queries.

1. **NL2NL** (Paraphrasing): The original natural language question is rewritten to improve clarity and model compatibility.
2. **NL2SQL** (SQL Generation): The paraphrased question and schema are used to generate a SQL query using a decoder-based LLM.

This project is built on top of the [Spider dataset](https://yale-lily.github.io/spider/) and uses the `vllm` inference engine for efficient LLM serving.

---

## Overall Input

- **Dataset**: JSON files from the Spider dataset (`train_spider.json`, `dev.json`, etc.)
- **Format**: Each row includes a natural language question, its associated SQL, and the schema (pipe-separated).
- **Location**: All inputs are located under `data/raw/` and processed to `data/processed/`.

---

##  Overall Output

- **Main output**: `results.csv` (in `results/`)
- **Contains**:
  - Original question
  - Paraphrased question
  - Ground truth SQL
  - Generated SQL
  - attempts
  - DB ID
  - attempts


---

##  How to Run

```bash
python main.py
```
---

##  Environment & Dependencies

-Python: 3.12.9
-Conda version: 4.8.2
-GPU: P100 2x 16GB (peak usage ~26GB)
-Runtime: ~4 hours for full run
-Requirements: See requirements.txt

---