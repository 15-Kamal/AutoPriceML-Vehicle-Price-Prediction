import streamlit as st
import pandas as pd
import joblib

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AutoPrice ML", page_icon="🏎️", layout="centered")

# --- 2. LOAD PIPELINE & EXTRACT CATEGORIES ---
@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load('autopriceml_pipeline.pkl')
        
        # MASSIVE PORTFOLIO FLEX: Extracting the exact training categories directly from the AI
        # 'preprocessor' -> 'cat' (the OneHotEncoder) -> categories_
        ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
        
        # ohe.categories_ returns a list of arrays in the order we trained them (Make, Model, Transmission)
        makes = ohe.categories_[0].tolist()
        models = ohe.categories_[1].tolist()
        transmissions = ohe.categories_[2].tolist()
        
        return pipeline, makes, models, transmissions
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None, [], [], []

pipeline, available_makes, available_models, available_transmissions = load_pipeline()

# --- 3. THE WEB INTERFACE ---
st.title("🏎️ AutoPrice: AI Vehicle Valuation")
st.write("Enter the specifications of a vehicle to get an instant, AI-driven market price estimate.")
st.divider()

if pipeline is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vehicle Details")
        # Now using dynamic dropdowns! Impossible to misspell or get case-sensitivity wrong.
        make = st.selectbox("Vehicle Make", available_makes)
        model = st.selectbox("Vehicle Model", available_models)
        transmission = st.selectbox("Transmission", available_transmissions)
        
    with col2:
        st.subheader("Age & Usage")
        year = st.slider("Year", min_value=2000, max_value=2024, value=2018)
        mileage = st.number_input("Mileage", min_value=0, max_value=300000, value=60000, step=1000)
        
    st.divider()

    # --- 4. PREDICTION LOGIC ---
    if st.button("Calculate Market Price", type="primary", use_container_width=True):
        with st.spinner("Analyzing market data..."):
            
            input_data = pd.DataFrame([{
                'make': make,
                'model': model,
                'year': year,
                'mileage': mileage,
                'transmission': transmission
            }])
            
            predicted_price = pipeline.predict(input_data)[0]
            
            # --- 5. DISPLAY RESULTS ---
            st.success("Valuation Complete!")
            st.metric(label="Estimated Market Value", value=f"${predicted_price:,.2f}")