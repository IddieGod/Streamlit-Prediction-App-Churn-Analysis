import streamlit as st

class History:
    def __init__(self):
        self.history = []

    def add_entry(self, entry):
        self.history.append(entry)

    def display_history(self):
        st.header("History")
        for i, entry in enumerate(self.history):
            st.write(f"{i+1}. {entry}")

# Create a shared instance of the History class
history = History()

# Home page
if st.button("Home"):
    history.add_entry("Visited Home page")

# Login page
if st.button("Login"):
    history.add_entry("Logged in")

# Upload dataset
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    history.add_entry("Dataset uploaded")

# Prediction
if st.button("Predict"):
    # Perform prediction
    prediction_result = predict_function()  # Call your prediction function
    history.add_entry(f"Prediction: {prediction_result}")

# View dashboards
if st.button("Dashboards"):
    history.add_entry("Viewed dashboards")

# Display history
history.display_history()

# Go back to previous points in history
go_to_history = st.selectbox("Go to history entry", [i+1 for i in range(len(history.history))])
if st.button("Go"):
    st.experimental_rerun(go_to_history)
