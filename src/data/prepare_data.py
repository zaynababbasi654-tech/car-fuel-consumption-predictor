import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "raw" / "Fuel_Consumption_2000-2022.csv"

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data types
print("\nData types:")
print(df.dtypes)

# Basic statistics
print("\nStatistics:")
print(df.describe())

# Save cleaned dataset
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "fuel_cleaned.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaned dataset saved successfully!")
print(OUTPUT_FILE)
    