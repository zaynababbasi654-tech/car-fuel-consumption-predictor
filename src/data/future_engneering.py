import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "fuel_cleaned.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "fuel_features.csv"

df = pd.read_csv(INPUT_FILE)

# Engine size × cylinders
df["engine_cylinder_interaction"] = (
    df["engine_size"] * df["cylinders"]
)

# Engine size per cylinder
df["engine_per_cylinder"] = (
    df["engine_size"] / df["cylinders"]
)

# Vehicle age
df["vehicle_age"] = 2022 - df["year"]

# Average fuel consumption
df["avg_fuel_consumption"] = (
    df["fuel_consumption_city_l_100_km"]
    + df["fuel_consumption_hwy_l_100_km"]
) / 2

# Save feature-engineered dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Feature engineering completed!")
print("New shape:", df.shape)
print("Saved to:", OUTPUT_FILE)