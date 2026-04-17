import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model + columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("🏠 Bengaluru House Price Predictor")

# ------------------ LOAD DATA FOR LOCATIONS ------------------
data = pd.read_csv("cleaned_data.csv")
locations = sorted(data['location'].unique())

# ------------------ INPUTS ------------------
location = st.selectbox("Location", locations)
sqft = st.number_input("Total Sqft", min_value=300, max_value=10000, step=50)
bath = st.number_input("Bathrooms", min_value=1, max_value=10)
bhk = st.number_input("BHK", min_value=1, max_value=10)

# ------------------ PREDICTION ------------------n
def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(columns))
    
    # numeric
    x[columns.get_loc("total_sqft")] = sqft
    x[columns.get_loc("bath")] = bath
    x[columns.get_loc("bhk")] = bhk

    # location
    loc_col = "location_" + location
    if loc_col in columns:
        x[columns.get_loc(loc_col)] = 1

    return model.predict([x])[0]

# ------------------ BUTTON ------------------
if st.button("Predict Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"💰 Estimated Price: ₹ {round(price,2)} Lakhs")