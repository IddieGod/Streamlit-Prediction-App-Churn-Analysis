import streamlit as st

def welcome_page():
    st.title("Welcome to Telecom Customer Churn Predictor App")
    st.write("""
    The Telecom Customer Churn Predictor App helps you predict the churn rate of customers in a telecom company.
    
    ## About
    The churn rate, also known as the rate of attrition or customer churn, is the rate at which customers stop doing business with a company.
    
    ## Log In
    Please log in to continue:
    """)
    
    # Username and password input fields
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    # Log in button
    if st.button("Log In"):
        # Perform authentication here
        # Replace this with your authentication logic
        if username == "admin" and password == "churn":
            st.success("Logged in successfully!")
            # Set session state to indicate the user is logged in
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password. Please try again.")

def main():
    welcome_page()

if __name__ == "__main__":
    main()

