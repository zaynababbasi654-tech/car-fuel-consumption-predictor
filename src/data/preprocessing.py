import pandas as pd
from pathlib import Path

# File paths
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA = BASE_DIR / "data" / "raw" / "Fuel_Consumption_2000-2022.csv"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "fuel_cleaned.csv"

# Load dataset
df = pd.read_csv(RAW_DATA)

print("Original shape:", df.shape)

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

print("After cleaning:", df.shape)

# Create processed folder if it doesn't exist
PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)

# Save cleaned dataset
df.to_csv(PROCESSED_DATA, index=False)

print("Cleaned dataset saved successfully!")
print("Saved at:", PROCESSED_DATA)