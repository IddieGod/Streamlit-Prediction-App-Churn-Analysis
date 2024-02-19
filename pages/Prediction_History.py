import streamlit as st
import pandas as pd

def main():
    st.title("Prediction History")

    # Sample prediction history data
    prediction_data = {
        "Timestamp": ["2024-02-20 10:00:00", "2024-02-19 15:30:00"],
        "Predicted Outcome": ["Churn", "No Churn"],
        "Confidence Level": [0.75, 0.85]
    }

    df_prediction = pd.DataFrame(prediction_data)

    st.write(df_prediction)

if __name__ == "__main__":
    main()
