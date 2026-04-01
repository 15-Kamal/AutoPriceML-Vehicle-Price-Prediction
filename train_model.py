import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import time
import joblib

# --- 1. Load the Data ---
print("1. Loading machine-learning-ready dataset...")
df = pd.read_csv('ml_ready_dataset.csv')

# --- 2. Separate Features (X) and Target (y) ---
# X is everything EXCEPT the price. y is ONLY the price.
print("2. Separating features (X) and target (y)...")
X = df.drop('price', axis=1)
y = df['price']

# --- 3. Split into Train and Test Sets ---
print("3. Splitting data into Training (80%) and Testing (20%)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Train the AI (Random Forest Regressor) ---
print("\n4. Training the Random Forest AI...")
print("(This might take a few seconds to process all 887 columns!)")
start_time = time.time()

# We use 100 'decision trees' (n_estimators) to vote on the final price
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print(f" Training complete in {time.time() - start_time:.2f} seconds.")

# --- 5. Test the AI and Evaluate ---
print("\n5. Testing the AI on unseen data...")
y_pred = model.predict(X_test)

# Calculate how wrong the AI was on average (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)

# Calculate how much of the pricing logic the AI figured out (R-Squared)
r2 = r2_score(y_test, y_pred)

print("\n--- AI PERFORMANCE METRICS ---")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"R-squared Score (R2): {r2:.4f}")

print("\n--- WHAT THIS MEANS ---")
print(f"- On average, the AI's price predictions are off by ${mae:,.0f}.")
print(f"- The AI has successfully learned {r2*100:.1f}% of the patterns that drive vehicle prices in this dataset.")

# --- 6. SAVE MODEL AND COLUMNS FOR STREAMLIT ---
print("\nSaving the model and column structure for the web app...")
joblib.dump(model, 'price_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl') # We must save the 887 column names!
print(" Saved 'price_model.pkl' and 'model_columns.pkl'!")