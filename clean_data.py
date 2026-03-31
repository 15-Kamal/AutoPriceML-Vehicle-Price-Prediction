import pandas as pd

# --- 1. Load the Data ---
print("1. Loading dataset...")
df = pd.read_csv('dataset.csv')

# --- 2. Drop Missing Target Variables ---
# We CANNOT train the model if the 'price' is missing.
print("\n2. Dropping rows with missing prices...")
initial_rows = len(df)
df = df.dropna(subset=['price'])
print(f"   Dropped {initial_rows - len(df)} rows. New total: {len(df)} vehicles.")

# --- 3. Drop Messy/Unnecessary Columns ---
# 'name' and 'description' are too complex for our model right now.
# We already have 'make', 'model', and 'trim' which do the same job.
columns_to_drop = ['name', 'description']
df = df.drop(columns=columns_to_drop)
print(f"\n3. Dropped text columns: {columns_to_drop}")

# --- 4. Fill Missing Numerical Values (Imputation) ---
print("\n4. Filling missing numerical values...")
# Fill missing mileage with the median mileage
df['mileage'] = df['mileage'].fillna(df['mileage'].median())
# Fill missing cylinders and doors with the most common number (mode)
df['cylinders'] = df['cylinders'].fillna(df['cylinders'].mode()[0])
df['doors'] = df['doors'].fillna(df['doors'].mode()[0])

# --- 5. Fill Missing Text Values ---
print("5. Filling missing categorical (text) values...")
# Fill any missing text with the word 'Unknown'
text_columns = df.select_dtypes(include=['object', 'string']).columns
for col in text_columns:
    df[col] = df[col].fillna('Unknown')

# --- 6. Final Check & Save ---
print("\n--- FINAL MISSING VALUES CHECK ---")
print(df.isnull().sum().max() == 0) # Should print True if everything is clean!

print("\n6. Saving cleaned data...")
df.to_csv('cleaned_dataset.csv', index=False)
print("Cleaned data saved as 'cleaned_dataset.csv'!")