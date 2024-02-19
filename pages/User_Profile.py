import streamlit as st

def main():
    st.title("User Profile")

    # Sample user profile data
    username = "JohnDoe"
    email = "johndoe@example.com"

    st.write(f"Username: {username}")
    st.write(f"Email: {email}")

    # Edit profile form
    st.subheader("Edit Profile")
    new_username = st.text_input("New Username", username)
    new_email = st.text_input("New Email", email)

    if st.button("Update Profile"):
        # Update profile logic goes here
        st.success("Profile updated successfully!")

if __name__ == "__main__":
    main()
