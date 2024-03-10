import streamlit as st

def main():
    st.title("Feedback")

    # Feedback form
    feedback = st.text_area("Please share your feedback", "")
    if st.button("Submit Feedback"):
        # Submit feedback logic goes here
        st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
