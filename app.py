import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #A0A0A0;
    font-size: 18px;
    margin-bottom: 30px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #6C63FF, #5A54E8);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    font-size: 18px;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #5A54E8, #6C63FF);
    transform: scale(1.02);
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1E1E1E;
    text-align: center;
    margin-top: 25px;
    border: 1px solid #333;
}

.price-text {
    color: #00FFAA;
    font-size: 32px;
    font-weight: bold;
}

.metric-card {
    background-color: #1A1D24;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #2D2D2D;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------- LOAD DATA ----------------
data = pd.read_csv("cleaned_data.csv")
locations = sorted(data['location'].unique())

# ---------------- HEADER ----------------
st.markdown(
    '<div class="title">🏠 Bengaluru House Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict house prices instantly using Machine Learning</div>',
    unsafe_allow_html=True
)

# ---------------- INPUT SECTION ----------------
with st.container():

    st.subheader("📍 Property Details")

    location = st.selectbox(
        "Select Location",
        locations
    )

    col1, col2 = st.columns(2)

    with col1:
        sqft = st.number_input(
            "📐 Total Sqft",
            min_value=300,
            max_value=10000,
            step=50,
            value=1200
        )

        bath = st.number_input(
            "🛁 Bathrooms",
            min_value=1,
            max_value=10,
            value=2
        )

    with col2:
        bhk = st.number_input(
            "🛏️ BHK",
            min_value=1,
            max_value=10,
            value=2
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Property Summary</h4>
            <p><b>{bhk} BHK</b></p>
            <p>{bath} Bathrooms</p>
            <p>{sqft} Sqft</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- PREDICTION FUNCTION ----------------
def predict_price(location, sqft, bath, bhk):

    x = np.zeros(len(columns))

    x[columns.get_loc("total_sqft")] = sqft
    x[columns.get_loc("bath")] = bath
    x[columns.get_loc("bhk")] = bhk

    loc_col = "location_" + location

    if loc_col in columns.values:
        x[columns.get_loc(loc_col)] = 1

    prediction = model.predict([x])[0]

    return prediction

# ---------------- BUTTON ----------------
if st.button("🚀 Predict Price"):

    price = predict_price(location, sqft, bath, bhk)

    st.markdown(f"""
    <div class="result-box">
        <h2>💰 Estimated House Price</h2>
        <div class="price-text">
            ₹ {round(price, 2)} Lakhs
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)

st.caption("Built with ❤️ using Streamlit & Machine Learning")