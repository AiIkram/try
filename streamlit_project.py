import gdown
import joblib
import os
import pandas as pd
import streamlit as st

# Google Drive File ID (from the URL)
file_id = "15u6GSqBo6YxEDELTVxuRX5ASPSRsSLhF"

# Path to save the model
MODEL_PATH = 'churning_model.pkl'

# Function to download the model from Google Drive
def download_model_from_drive(file_id, destination):
    gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", destination, quiet=False)

# Function to load the model
def load_model():
    if not os.path.exists(MODEL_PATH):
        # If the model is not found, download it from Google Drive
        download_model_from_drive(file_id, MODEL_PATH)
    return joblib.load(MODEL_PATH)
    

import os
if os.path.exists(MODEL_PATH):
    print("Model file exists")
else:
    print("Model file does not exist")

# Load the model
model = load_model()

# Ensure model is loaded
if model is None:
    st.error("Failed to load the model!")
else:
    st.success("Model loaded successfully!")

# Inspect the features the model expects
expected_features = list(model.feature_names_in_)

# Define the app
st.title('Customer Churn Prediction')

st.header('Enter Customer Details')

# Collect input features
inputs = {
    'REGION': st.number_input('Region', min_value=0.0, max_value=10.0, step=1.0, value=1.0),  # Add REGION
    'TENURE': st.slider('Tenure (months)', 0, 72, 13),
    'MONTANT': st.number_input('Montant', min_value=0.0, max_value=15350.0, step=1.0, value=3600.0),
    'FREQUENCE_RECH': st.number_input('Frequence Recherche', min_value=0.0, max_value=33.0, step=1.0, value=7.0),
    'REVENUE': st.number_input('Revenue', min_value=0.0, max_value=15824.0, step=1.0, value=1020.0),
    'ARPU_SEGMENT': st.number_input('ARPU Segment', min_value=0.0, max_value=5275.0, step=1.0, value=340.0),
    'FREQUENCE': st.number_input('Frequence', min_value=0.0, max_value=41.0, step=1.0, value=2.0),
    'DATA_VOLUME': st.number_input('Data Volume', min_value=0.0, max_value=8359.0, step=1.0, value=3366.45),
    'ON_NET': st.number_input('On Net', min_value=0.0, max_value=683.0, step=1.0, value=90.0),
    'ORANGE': st.number_input('Orange', min_value=0.0, max_value=223.0, step=1.0, value=46.0),
    'TIGO': st.number_input('Tigo', min_value=0.0, max_value=50.0, step=1.0, value=7.0),
    'ZONE1': st.number_input('Zone 1', min_value=0.0, max_value=8.0, step=0.1, value=8.0),
    'ZONE2': st.number_input('Zone 2', min_value=0.0, max_value=8.0, step=0.1, value=7.55),
    'MRG': st.number_input('MRG', min_value=0.0, max_value=62.0, step=1.0, value=0.0),
    'REGULARITY': st.number_input('Regularity', min_value=0.0, max_value=117.0, step=1.0, value=17.0),
    'TOP_PACK': st.number_input('Top Pack', min_value=0.0, max_value=22.0, step=1.0, value=22.0),
    'FREQ_TOP_PACK': st.number_input('Freq Top Pack', min_value=0.0, max_value=22.0, step=1.0, value=0.0),
}

# Create input DataFrame
input_df = pd.DataFrame([inputs])

# Reorder columns to match the model
try:
    input_df = input_df[expected_features]
except KeyError as e:
    st.error(f"Missing or mismatched features: {e}")
    st.stop()

# Predict and display results
if st.button('Predict'):
    try:
        prediction = model.predict(input_df)
        st.write('Prediction: Churn' if prediction[0] else 'Prediction: Not Churn')
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
