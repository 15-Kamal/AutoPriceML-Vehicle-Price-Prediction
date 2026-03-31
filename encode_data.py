import pandas as pd

# --- 1. Load the Cleaned Data ---
print("1. Loading cleaned dataset...")
df = pd.read_csv('cleaned_dataset.csv')

# --- 2. One-Hot Encoding ---
print("\n2. Translating text columns into numbers (One-Hot Encoding)...")

# pd.get_dummies automatically finds text columns and converts them to 1s and 0s
# drop_first=True is a statistical trick to prevent overlapping data (multicollinearity)
df_encoded = pd.get_dummies(df, drop_first=True)

print(f" Old feature count (before encoding): {df.shape[1]}")
print(f" New feature count (after encoding): {df_encoded.shape[1]}")

# --- 3. Save the ML-Ready Data ---
print("\n3. Saving the final machine-learning-ready dataset...")
df_encoded.to_csv('ml_ready_dataset.csv', index=False)

print(" Success! Saved as 'ml_ready_dataset.csv'.")