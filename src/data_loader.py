"""
data_loader.py
--------------
Loads the KPI dataset and identifies which columns are dimensions
(categorical) vs metrics (numeric).

This is the foundation — every other module depends on this.
"""

import os
import re
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION — Change this path to wherever your CSV lives
# ---------------------------------------------------------------------------
# When running locally in VS Code, this looks in the data/ folder
# When running in Colab, you'd change this to your Google Drive path
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_ecommerce_kpi_data.csv")
DATE_ALIASES = ["date", "timestamp", "datetime", "day", "event_date", "record_date"]

def clean_column_name(col: str) -> str:
    col = str(col).strip()
    col = re.sub(r"\s+", "_", col)
    col = col.lower()
    return col

def normalize_columns(df: pd.DataFrame):
    original_cols = df.columns.tolist()
    cleaned_cols = [clean_column_name(c) for c in original_cols]
    rename_map = dict(zip(original_cols, cleaned_cols))
    df = df.rename(columns=rename_map)
    return df, rename_map

def infer_date_column(df: pd.DataFrame, preferred_date_col: str = None):
    cols = list(df.columns)

    if preferred_date_col:
        preferred = clean_column_name(preferred_date_col)
        if preferred in cols:
            return preferred

    for alias in DATE_ALIASES:
        if alias in cols:
            return alias

    # fallback: first column that looks parseable as datetime
    for col in cols:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                return col
        except Exception:
            pass

    return None

def load_data(path=None):
    """
    Load the CSV file and return:
    - df: the DataFrame with 'date' parsed as datetime
    - dimension_keys: list of categorical columns (e.g., region, device_type)
    - metric_keys: list of numeric columns (e.g., revenue, orders)
    """
    if path is None:
        path = DATA_PATH

    df = pd.read_csv(path)
    df, rename_map = normalize_columns(df)

    date_col = infer_date_column(df, preferred_date_col)
    if date_col is None:
        raise ValueError(
            f"Could not infer a date column. Available columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()

    if date_col != "date":
        df = df.rename(columns={date_col: "date"})

    all_columns = df.columns.tolist()

    dimension_keys = [
        c for c in all_columns
        if c != "date" and not pd.api.types.is_numeric_dtype(df[c])
    ]

    metric_keys = [
        c for c in all_columns
        if c != "date" and pd.api.types.is_numeric_dtype(df[c])
    ]

    return {
        "df": df,
        "dimension_keys": dimension_keys,
        "metric_keys": metric_keys,
        "column_mapping": rename_map,
        "resolved_date_column": date_col,
    }


# ---------------------------------------------------------------------------
# If you run this file directly, it prints a summary of the dataset
# Useful for quick sanity checks: python src/data_loader.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = load_data()
    df = result["df"]
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Dimensions: {result['dimension_keys']}")
    print(f"Metrics: {result['metric_keys']}")
    print(f"Resolved date column: {result['resolved_date_column']}")
