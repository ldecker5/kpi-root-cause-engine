"""
generate_sample_data.py
-----------------------
Run this ONCE to create the sample CSV dataset.

Usage: python generate_sample_data.py

The dataset simulates an e-commerce company with:
- A revenue drop starting around April 20, 2024
- The drop is caused by: mobile conversion dropping + West region marketing cut
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Generate dates: Jan 1 to June 30, 2024
dates = pd.date_range("2024-01-01", "2024-06-30", freq="D")
regions = ["East", "West", "North", "South"]
device_types = ["Mobile", "Desktop"]

rows = []
for date in dates:
    for region in regions:
        for device in device_types:
            # Base values
            base_revenue = 5000
            base_orders = 100
            base_conversion = 0.045
            base_marketing = 3000
            base_sessions = 2200

            # Regional variation
            if region == "West":
                base_revenue *= 1.2
                base_marketing *= 1.3
            elif region == "East":
                base_revenue *= 1.1
            elif region == "South":
                base_revenue *= 0.85

            # Device variation
            if device == "Mobile":
                base_conversion *= 0.9
                base_sessions *= 1.3
            else:
                base_sessions *= 0.7

            # === THE PLANTED ANOMALY ===
            # Starting April 20, mobile conversion drops and West marketing gets cut
            anomaly_date = pd.Timestamp("2024-04-20")
            if date >= anomaly_date:
                if device == "Mobile":
                    base_conversion *= 0.77  # 23% drop in mobile conversion
                if region == "West":
                    base_marketing *= 0.65   # 35% cut in West marketing

                # Revenue impact from both factors
                if device == "Mobile" and region == "West":
                    base_revenue *= 0.70     # Hardest hit
                elif device == "Mobile":
                    base_revenue *= 0.85
                elif region == "West":
                    base_revenue *= 0.90

            # Add random noise
            revenue = max(0, base_revenue + np.random.normal(0, base_revenue * 0.08))
            orders = max(1, int(base_orders + np.random.normal(0, 12)))
            conversion = max(0.005, base_conversion + np.random.normal(0, 0.005))
            marketing = max(0, base_marketing + np.random.normal(0, base_marketing * 0.05))
            sessions = max(100, int(base_sessions + np.random.normal(0, 150)))

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "region": region,
                "device_type": device,
                "revenue": round(revenue, 2),
                "orders": orders,
                "conversion_rate": round(conversion, 4),
                "marketing_spend": round(marketing, 2),
                "sessions": sessions,
            })

df = pd.DataFrame(rows)

# Save to data/ folder
output_path = os.path.join(os.path.dirname(__file__), "data", "sample_ecommerce_kpi_data.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Created dataset: {output_path}")
print(f"   Rows: {len(df)}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
print(f"   Columns: {df.columns.tolist()}")
print(f"\n   Anomaly planted: April 20+ → Mobile conversion -23%, West marketing -35%")
