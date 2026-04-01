import streamlit as st
import pandas as pd
import joblib

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AutoPrice ML", page_icon="🚗", layout="centered")

# --- 2. LOAD MODEL & COLUMNS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('price_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    except FileNotFoundError:
        st.error("Missing .pkl files! Did you run your training script to save them?")
        return None, None

model, model_columns = load_assets()

# --- 3. THE WEB INTERFACE ---
st.title(" AutoPrice: AI Vehicle Valuation")
st.write("Enter the specifications of a vehicle to get an instant, AI-driven market price estimate.")
st.divider()

if model is not None and model_columns is not None:
    
    # Extract unique car brands (makes) from our 887 columns
    available_makes = [col.replace('make_', '') for col in model_columns if col.startswith('make_')]
    available_makes.sort()

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vehicle Specs")
        make = st.selectbox("Vehicle Make", available_makes)
        year = st.slider("Year", min_value=2010, max_value=2024, value=2020)
        mileage = st.number_input("Mileage", min_value=0, max_value=250000, value=45000, step=1000)
    
    with col2:
        st.subheader("Engine & Body")
        cylinders = st.selectbox("Engine Cylinders", [4, 6, 8])
        doors = st.selectbox("Doors", [2, 4])
        
    st.divider()

    # --- 4. PREDICTION LOGIC ---
    if st.button(" Calculate Market Price", type="primary", use_container_width=True):
        with st.spinner("Analyzing market data..."):
            
            # Step A: Create a blank dataframe with all 887 columns filled with 0s
            input_df = pd.DataFrame(0, index=[0], columns=model_columns)
            
            # Step B: Insert the numerical values the user provided
            input_df['year'] = year
            input_df['mileage'] = mileage
            input_df['cylinders'] = cylinders
            input_df['doors'] = doors
            
            # Step C: The One-Hot Encoding Trick
            make_column = f'make_{make}'
            if make_column in input_df.columns:
                input_df[make_column] = 1
                
            # Step D: Predict the price!
            predicted_price = model.predict(input_df)[0]
            
            # --- 5. DISPLAY RESULTS ---
            st.success("Valuation Complete!")
            st.metric(label="Estimated Market Value", value=f"${predicted_price:,.2f}")
            st.caption("Disclaimer: This is an AI-generated estimate based on historical data. Real-world prices may vary due to vehicle condition, accident history, and local market factors.")