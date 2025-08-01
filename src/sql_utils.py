# sql_utils.py
# Utility functions for database connection, query execution, and result comparison.

import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

def check_db_exists(db_path):
    """Raise FileNotFoundError if the database file does not exist."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

def connect_to_db(db_path):
    """Connect to the SQLite database at db_path."""
    try:
        return sqlite3.connect(db_path)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to database: {db_path}. Error: {e}")

def check_table_exists(conn, table_name):
    """Raise ValueError if the table does not exist in the database."""
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name=?;"
    result = conn.execute(query, (table_name,)).fetchone()
    if not result:
        raise ValueError(f"Table '{table_name}' does not exist.")

def run_query(conn, query):
    """Run a SQL query and return the result as a DataFrame. Raise ValueError on failure."""
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        raise ValueError(f"Query failed: {query}. Error: {e}")


def compare_results(df1, df2):
    """Compare two DataFrames for equality, ignoring row/column order and NaNs."""
    if df1.shape != df2.shape:
        return False

    #print(df1)
    #print("===")
    #print(df2)

    df1_vals = df1.values
    df2_vals = df2.values

    if np.array_equal(df1_vals, df2_vals):
        return True

    if (df1_vals == df2_vals).all():
        return True
    else:
        def row_to_str(row):
            return "|".join(str('' if pd.isna(x) else x) for x in row)
             
        a_str = "".join(sorted([row_to_str(row) for row in df1_vals]))
        b_str = "".join(sorted([row_to_str(row) for row in df2_vals]))

        if a_str == b_str:
            return True
        try:
            return sorted(a_str) == sorted(b_str)
        except Exception:
            return False

def run_all(dbname, query1, query2):
    """Compare two SQL queries on the same database. Returns True if results match, else False."""
    if query1 == query2:
        return True
    db_path = project_root/"data"/f"database/{dbname}/{dbname}.sqlite"
    check_db_exists(db_path)
    try:
        conn = connect_to_db(db_path)
        df1 = run_query(conn, query1)
        df2 = run_query(conn, query2)
        conn.close()
        return compare_results(df1, df2)
    except Exception as e:
        print(f"[run_all] Error: {e}")
        return False

def generate_simplified_schema(db_name):
    """Generate a simplified schema description for the given database.
    Returns a string representation of the schema."""
    simplified_schema = [f"Database: {db_name}\n"]
    try:
        db_path = project_root/"data"/f"database/{db_name}/{db_name}.sqlite"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for (table_name,) in tables:
                simplified_schema.append(f"Table: {table_name}")
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                for col in columns:
                    col_name = col[1]
                    col_type = col[2].lower()
                    is_pk = bool(col[5])
                    pk_text = " (primary key)" if is_pk else ""
                    simplified_schema.append(f"- {col_name}: {col_type}{pk_text}")
                simplified_schema.append("")  # Blank line between tables
        return "\n".join(simplified_schema)
    except Exception as e:
        print(f"Error extracting schema from {db_name}: {e}")
        return None


if __name__ == "__main__":
    # Example usage and test
    sample_dbname = "cre_Doc_Template_Mgt"
    query1 = "SELECT template_type_code ,MIN(version) AS min_version FROM Templates GROUP BY template_type_code;. Error: Execution failed on sql 'SELECT template_type_code ,MIN(version) AS min_version FROM Templates GROUP BY template_type_code;"
    query2 = "SELECT version_number , template_type_code FROM templates ORDER BY version_number ASC LIMIT 1;"
    print(run_all(sample_dbname, query1, query2))
    schema= generate_simplified_schema(sample_dbname)
    print(schema)