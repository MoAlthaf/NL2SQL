## NL2NL & NL2SQL Paraphrasing and SQL Generation Pipeline

This project provides a two-stage pipeline for:

1. **Paraphrasing natural language questions** about databases (NL2NL)
2. **Generating and verifying SQL queries** from those questions (NL2SQL)

### How it works

- **Stage 1 (NL2NL):**
  - `nl2nl.py` takes a dataset of prompts and paraphrases each question using a large language model (LLM), ensuring the meaning and SQL intent are preserved.
  - Output: `outputs/output_paraphrased.csv` with paraphrased questions.

- **Stage 2 (NL2SQL):**
  - `nl2sql.py` takes the paraphrased questions and generates SQL queries using another LLM.
  - It then verifies the generated SQL against ground truth using actual database execution and result comparison.
  - Output: `results.csv` with generated SQL, match status, and attempt count.

### How to run

1. **Prepare your environment:**
   - Install dependencies from `requirements.txt` (Python 3.8+ recommended).
   - Ensure you have access to a GPU (minimum: 24GB VRAM).
   - Place your database files in the `database/` directory as required.

2. **Run the pipeline:**
   - **Option 1: Run each stage manually**
     - Step 1: Paraphrase questions
       ```bash
       python nl2nl.py
       ```
     - Step 2: Generate and verify SQL
       ```bash
       python nl2sql.py
       ```
   - **Option 2: Run the full pipeline with one command**
     - This will run both stages and handle dataset setup and GPU cache clearing automatically:
       ```bash
       python main.py
       ```

### Design

- The pipeline is modular: you can run each stage independently.
- Both scripts use LLMs (configurable at the top of each script) and require a GPU for efficient inference.
- Results are saved as CSV files for easy analysis.

### Minimum GPU Requirements

- At least one NVIDIA GPU with 16GB VRAM (e.g., V100, A100, or better) is recommended for smooth LLM inference.

---
For more details, see comments in each script.
