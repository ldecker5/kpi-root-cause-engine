"""
data_loader.py
--------------
Loads the KPI dataset and identifies which columns are dimensions
(categorical) vs metrics (numeric).

This is the foundation — every other module depends on this.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION — Change this path to wherever your CSV lives
# ---------------------------------------------------------------------------
# When running locally in VS Code, this looks in the data/ folder
# When running in Colab, you'd change this to your Google Drive path
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_ecommerce_kpi_data.csv")


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
    df["date"] = pd.to_datetime(df["date"])

    all_columns = df.columns.tolist()

    # Dimensions = non-date, non-numeric columns (like "region", "device_type")
    dimension_keys = [
        c for c in all_columns
        if c != "date" and not pd.api.types.is_numeric_dtype(df[c])
    ]

    # Metrics = non-date, numeric columns (like "revenue", "orders", "conversion_rate")
    metric_keys = [
        c for c in all_columns
        if c != "date" and pd.api.types.is_numeric_dtype(df[c])
    ]

    return df, dimension_keys, metric_keys


# ---------------------------------------------------------------------------
# If you run this file directly, it prints a summary of the dataset
# Useful for quick sanity checks: python src/data_loader.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df, dims, metrics = load_data()
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Dimensions: {dims}")
    print(f"Metrics: {metrics}")
    print(f"\nFirst 5 rows:")
    print(df.head())
