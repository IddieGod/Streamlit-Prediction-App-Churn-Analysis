import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import matplotlib.pyplot as plt
import seaborn as sns


# Function to preprocess the uploaded dataset
def preprocess_uploaded_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Drop any irrelevant columns or perform any necessary preprocessing steps
        # For example:
        # df.drop(['Column1', 'Column2'], axis=1, inplace=True)
        return df
    else:
        return None

# Function to download CSV file
def download_dataset(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="uploaded_data.csv">Download CSV File</a>'
    return href

# Function to display visualizations of uploaded data
def display_visualizations(df):
    st.header("Data Visualizations")
    st.subheader("Pairplot")
    sns.pairplot(df)
    st.pyplot()

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    st.pyplot()


# Sidebar: File uploader for user to upload their own dataset
uploaded_file = st.sidebar.file_uploader("Upload your own dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Preprocess the uploaded dataset
    uploaded_df = preprocess_uploaded_data(uploaded_file)

    if uploaded_df is not None:
        st.sidebar.markdown(download_dataset(uploaded_df), unsafe_allow_html=True)

        # Display the uploaded dataset
        st.header('Uploaded Dataset Overview')
        st.write('Data Dimension: ' + str(uploaded_df.shape[0]) + ' rows and ' + str(uploaded_df.shape[1]) + ' columns.')
        st.dataframe(uploaded_df)

        # Display visualizations of the uploaded data
        display_visualizations(uploaded_df)
