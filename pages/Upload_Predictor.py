import streamlit as st
import pandas as pd
import numpy as np
import base64
import joblib
import os

st.set_page_config(page_title='Telecom Churn Prediction', page_icon=':Data:')


# Function to load the model and encoder
def load_model_and_encoder():
    model_path = 'models/finished_pipeline.joblib'
    encoder_path = 'models/encoder.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        st.error("Model or encoder not found. Please make sure the model and encoder files exist.")
        return None, None
    
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

# Function to preprocess the uploaded dataset
def preprocess_uploaded_data(uploaded_file, encoder):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Drop any irrelevant columns or perform any necessary preprocessing steps
        # For example:
        # df.drop(['Column1', 'Column2'], axis=1, inplace=True)
        # Encode categorical variables
        df_encoded = encoder.transform(df)
        return df_encoded
    else:
        return None

# Sidebar: File uploader for user to upload their own dataset
uploaded_file = st.sidebar.file_uploader("Upload your own dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load model and encoder
    model, encoder = load_model_and_encoder()
    
    if model is not None and encoder is not None:
        # Preprocess the uploaded dataset
        uploaded_df = preprocess_uploaded_data(uploaded_file, encoder)

        if uploaded_df is not None:
            st.sidebar.markdown(download_dataset(uploaded_df), unsafe_allow_html=True)

            # Display the uploaded dataset
            st.header('Uploaded Dataset Overview')
            st.write('Data Dimension: ' + str(uploaded_df.shape[0]) + ' rows and ' + str(uploaded_df.shape[1]) + ' columns.')
            st.dataframe(uploaded_df)

            # Perform prediction using the uploaded dataset
            if st.sidebar.button('Predict Churn'):
                # Perform prediction using the uploaded dataset
                predictions = model.predict(uploaded_df)
                # Display the predictions
                st.write(predictions)
