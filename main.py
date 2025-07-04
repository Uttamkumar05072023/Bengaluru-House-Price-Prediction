import streamlit as st
import pandas as pd
import joblib

data = pd.read_csv("Cleaned_data.csv")
model = joblib.load("House_model.joblib")

# Custom CSS for background and card style
st.markdown(
    """
    <style>
    root {
        color-scheme: light dark; /* Required for light-dark() to work */
    }
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        border-radius: 8px;
        width: 100%;
    }
    .stButton>button>div>p{
        font-size: 20px;
        font-weight: 900;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title('🏠 Bengaluru House Price Predictor')
st.sidebar.markdown('''
This app predicts house prices in Bengaluru using a trained ML model.\

- Select the location, area, BHK, and bathrooms.\
- Click **Predict Price** to see the estimated value.
''')
st.sidebar.info('Made with ❤️ using Streamlit')

# Main Title
st.markdown('<h1 style="color: light-dark(black, white); font-size: 40px; text-align:center; font-weight:900;">Bengaluru House Price Prediction 🏡</h1>', unsafe_allow_html=True)
st.write("")

# Input fields in columns
col1, col2 = st.columns(2)
with col1:
    location = st.selectbox('Select Location', sorted(data['location'].unique()))
    bath = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
with col2:
    total_sqft = st.number_input('Total Square Feet', min_value=100.0, max_value=10000.0, value=1000.0)
    bhk = st.number_input('Number of BHK', min_value=1, max_value=10, value=2)

st.write("")

if st.button('🔍 Predict Price'):
    input_df = pd.DataFrame([[location, total_sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = model.predict(input_df)[0]
    st.markdown(f'<div style="background-color:#d4efdf; padding: 20px; border-radius: 10px; text-align:center; font-size: 28px; color:#145a32;">\n<b>Estimated Price:</b> ₹ {prediction:.2f} Lakhs</div>', unsafe_allow_html=True)

st.markdown('<hr style="border:1px solid #bbb;">', unsafe_allow_html=True)
st.markdown('<div style="text-align:center; color:#888;">© 2024 Bengaluru House Price Predictor | Powered by Streamlit</div>', unsafe_allow_html=True)
