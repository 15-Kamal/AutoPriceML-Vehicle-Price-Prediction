import pandas as pd
import os

# --- 1. Load the Data ---
FILE_PATH = 'dataset.csv'

if not os.path.exists(FILE_PATH):
    print(f"Error: Could not find '{FILE_PATH}'. Make sure it is in your project folder.")
    exit()

print("Loading vehicle dataset...\n")
df = pd.read_csv(FILE_PATH)

# --- 2. Dataset Overview ---
print("--- DATASET SHAPE ---")
print(f"Total Vehicles (Rows): {df.shape[0]}")
print(f"Total Features (Columns): {df.shape[1]}\n")

print("--- FIRST 5 ROWS ---")
print(df.head(), "\n")

# --- 3. Check for Missing Data ---
print("--- MISSING VALUES ---")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0]) # Only print columns that actually have missing data
if missing_data.sum() == 0:
    print("No missing data found! The dataset is clean.\n")

# --- 4. Data Types (Numbers vs Text) ---
print("\n--- DATA TYPES ---")
print(df.dtypes)