# Create App
import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# Load the trained model and scaler
model = load('best_trained_model.joblib')
scaler = load('robust_scaler.joblib')

# Load feature names from df_reduced to dynamically create input fields
feature_df = load('df_reduced.joblib') 

feature_names = feature_df.columns

st.title("Golden Apple Snail Prediction App")

# Choose between single and batch prediction
option = st.sidebar.selectbox(
    "Choose Prediction Method",
    ("Single Prediction", "Batch Prediction (Upload File)")
)

# Function to make predictions
def make_prediction(input_data):
    # Convert input data into a DataFrame and scale it
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    predictions = model.predict(input_scaled)
    return predictions[0]

# Single Prediction
if option == "Single Prediction":
    st.header("Enter Values for Each Feature")

    # Collect user input for each feature
    input_data = {}
    for feature in feature_names:
        input_data[feature] = st.number_input(f"{feature}", value=0.0000, format="%.3f")


    # Predict button
    if st.button("Predict"):
        prediction = make_prediction(input_data)
        baby, juvenile, adult = prediction  # Unpack prediction values
        if baby < 0:
            baby = 0
        if juvenile < 0:
            juvenile = 0
        if adult < 0:
            adult = 0
        st.success(f"Predicted Values:\n - Baby Snails: {np.round(baby)}\n - Juvenile Snails: {np.round(juvenile)}\n - Adult Snails: {np.round(adult)}\n - TOTAL: {np.round(np.round(baby)+np.round(juvenile)+np.round(adult))}")

# Batch Prediction via File Upload
elif option == "Batch Prediction (Upload File)":
    st.header("Upload an Excel or CSV File")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['csv', 'xlsx'])
    
    if uploaded_file:
        # Read the file based on its extension
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Check if file contains required features
        if all(feature in data.columns for feature in feature_names):
            # Scale the features
            data_scaled = scaler.transform(data[feature_names])

            # Make predictions
            predictions = model.predict(data_scaled)
            data[['BABY', 'JUVENILE', 'ADULT']] = predictions

            data[['BABY', 'JUVENILE', 'ADULT']] = np.round(data[['BABY', 'JUVENILE', 'ADULT']])
            # Replace values less than 0 with 0 in the specified columns
            data[['BABY', 'JUVENILE', 'ADULT']] = data[['BABY', 'JUVENILE', 'ADULT']].clip(lower=0)
            data['TOTAL'] = data[['BABY', 'JUVENILE', 'ADULT']].sum(axis=1)
            
            st.write("Predictions:")
            st.dataframe(data)

            # Download predictions as CSV
            csv_data = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_data,
                file_name='predictions.csv',
                mime='text/csv'
            )
        else:
            st.error("Uploaded file does not contain the required features.")
