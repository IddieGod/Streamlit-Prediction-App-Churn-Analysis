import streamlit as st
import os
import yaml
from streamlit_authenticator import Authenticate
from authenticate_files.validator import Validator
from authenticate_files.utils import generate_random_pw
from session_state import SessionState
import uuid
from authenticate_files.hasher import Hasher

# Function to generate unique key
def generate_key():
    return str(uuid.uuid4())

# Load configuration from YAML file
with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

# Set the favicon
st.set_page_config(page_title="IDG Customer Churn Predictor App", page_icon="🏠")

# Check if the user is already logged in
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Initialize form state if it doesn't exist
if 'form_state' not in st.session_state:
    st.session_state['form_state'] = {
        "username": "",
        "password": "",
        "new_username": "",
        "new_password": "",
        "confirm_password": ""
    }

# Home page function
def home_page():
    st.subheader("Home")
    st.write("This is a Streamlit app for predicting churn.")
    image_filename = "home.png"
    home_image_path = os.path.join("images", image_filename)
    st.image(home_image_path, use_column_width=True)

# Logout function
def logout():
    st.session_state['authenticated'] = False

# Main function
def main():
    if st.session_state['authenticated']:
        # Display the home page if the user is authenticated
        home_page()
        logout_button = st.button("Logout")
        if logout_button:
            logout()
    else:
        # Display the login and sign-up forms if the user is not authenticated
        col1, col2 = st.columns(2)

        with col1:
            st.title("IDG Customer Churn Predictor App")
            st.image("images/login.png", width=250)
            username = st.text_input("Username", key=f"login_username_input_{generate_key()}", value=st.session_state['form_state']["username"])
            password = st.text_input("Password", type="password", key=f"login_password_input_{generate_key()}", value=st.session_state['form_state']["password"])

        with col2:
            st.write("Don't have an account? Create one below.")
            new_username = st.text_input("New Username", key=f"new_username_input_{generate_key()}", value=st.session_state['form_state']["new_username"])
            new_password = st.text_input("New Password", type="password", key=f"new_password_input_{generate_key()}", value=st.session_state['form_state']["new_password"])
            confirm_password = st.text_input("Confirm Password", type="password", key=f"confirm_password_input_{generate_key()}", value=st.session_state['form_state']["confirm_password"])

        create_account_button_key = generate_key()
        create_account_button = st.button("Create Account", key=create_account_button_key)

        login_button_key = generate_key()
        login_button = st.button("Login", key=login_button_key)

        if create_account_button:
            st.session_state['form_state'] = {
                "username": username,
                "password": password,
                "new_username": new_username,
                "new_password": new_password,
                "confirm_password": confirm_password
            }
            validator = Validator()
            if not validator.validate_username(new_username):
                st.error("Invalid username format. Please use only alphanumeric characters, underscores, and hyphens.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                # Create a Hasher instance with the new password
                hasher = Hasher([new_password])
                # Generate the hashed password
                hashed_password = hasher.generate()[0]
                # Create the new account with the hashed password
                authenticator.create_account(new_username, hashed_password)
                st.success("Account created successfully. You can now log in.")

        if login_button:
            st.session_state['form_state'] = {
                "username": username,
                "password": password,
                "new_username": new_username,
                "new_password": new_password,
                "confirm_password": confirm_password
            }
            authenticator = Authenticate(config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days'], config['preauthorized'])
            try:
                auth_result = authenticator.login(username, password)
                if auth_result[1]:
                    st.session_state['authenticated'] = True
                    main()  # Recursive call to the main function
                else:
                    st.error("Login failed: Incorrect username or password.")
            except Exception as e:
                st.error(f"Login failed: {str(e)}")

if __name__ == "__main__":
    main()
