
# ==============================
# 1. Import Required Libraries
# ==============================

import joblib
import pandas as pd
import streamlit as st

# ==============================
# 2. Load Trained Model
# ==============================

model = joblib.load("RandomForest_tuned.pkl")

# ==============================
# 3. App Title & UI Elements
# ==============================

st.title("Car Selling Price Prediction")
st.header("Upload Data or Enter Manually")
st.subheader("Manual Feature Input")

# ==============================
# 4. Model Feature List
# ==============================

# These must match the training features exactly
features = [
    'year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
    'owner_First Owner', 'owner_Fourth & Above Owner',
    'owner_Second Owner', 'owner_Test Drive Car',
    'owner_Third Owner', 'fuel_CNG', 'fuel_Diesel',
    'fuel_LPG', 'fuel_Petrol', 'seller_type_Dealer',
    'seller_type_Individual', 'seller_type_Trustmark Dealer',
    'transmission_Automatic', 'transmission_Manual'
]

# ==============================
# 5. Collect User Inputs
# ==============================

inputs = []

for feature in features:
    value = st.number_input(f"Enter {feature}", value=0.0)
    inputs.append(value)

# ==============================
# 6. CSV Upload Option (Preview Only)
# ==============================

uploaded_file = st.file_uploader("Upload CSV file (optional)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df)

# ==============================
# 7. Prediction Section
# ==============================

if st.button("Predict Price"):

    prediction = model.predict([inputs])
    predicted_price = prediction[0]

    st.success(f"Predicted Selling Price: ₹ {predicted_price:,.2f}")
