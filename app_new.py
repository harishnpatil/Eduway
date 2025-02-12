import streamlit as st
from recommendation_model import generate_learning_path
import pandas as pd

# Set the title of the app
st.title("Your Virtual Learning Assistant")

# Add an about section
st.write("""
## About
Welcome to your virtual learning assistant! We are here to help you complete your course and achieve your learning goals. 
Fill out the form below to get started, and we'll generate a personalized learning path just for you.
""")

# Create a form to collect user information
with st.form("user_info_form"):
    st.write("### User Information")
    name = st.text_input("Name")
    email = st.text_input("Email")
    age = st.number_input("Age", min_value=0, max_value=120, value=20)
    education_level = st.selectbox("Education Level", ["High School", "Undergraduate", "Graduate", "Other"])
    
    st.write("### Goals to Learn")
    goals = st.text_area("What are your learning goals?")
    
    # Add a submit button
    submitted = st.form_submit_button("Submit")

# Process the form submission
if submitted:
    # Validate form inputs
    if not name or not email or not goals:
        st.error("Please fill out all required fields.")
    else:
        st.success("Form submitted successfully!")
        
        # Generate recommendations based on the user's goals
        recommendations = generate_learning_path(goals)
        
        # Display the recommendations in a tree structure
        st.write("### Personalized Learning Path")
        st.write(recommendations)

        # Optionally, save user information to a DataFrame
        user_info = pd.DataFrame({
            "Name": [name],
            "Email": [email],
            "Age": [age],
            "Education Level": [education_level],
            "Goals": [goals]
        })
        st.write("### User Information")
        st.write(user_info)

# Optionally, you can add a section to display the user information DataFrame
# st.write("### User Information")
# st.write(user_info)