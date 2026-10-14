# streamlit UI

import pickle
import pandas as pd
import streamlit as st

# page setup

st.set_page_config(page_title="Loan Approval Classification",page_icon = "assets/logo.png")

# sidebar setup
with st.sidebar:
    st.title("Loan Approval Classification")
    st.image("assets/logo.png")


# load the model ( model deserialization)

with open("best_model/best_model_pipeline.pkl","rb") as file:
    best_pipeline = pickle.load(file)
    

df = pd.read_csv("datasets/cleaned_data.csv")


# user input fields
col1, col2, col3 , col4= st.columns(4, border=True)

with col1:
    person_age = st.slider("Age :", min_value=1, max_value=100)
    person_emp_exp = st.slider("Experience :", min_value=0, max_value=50)
    person_income = st.slider("Income :", min_value= 10000,max_value = 1000000)
    loan_amnt = st.slider("Loan Amount :", min_value=15000,max_value=1000000)

with col2:
    person_gender = st.radio("Gender :", options=df["person_gender"].unique(), horizontal=True)
    loan_int_rate = st.slider("Loan Interest Rate :", min_value=0.0, max_value=30.0)
    loan_percent_income = st.slider("Loan Percent Income :", min_value=0.0, max_value=100.0)
    cb_person_cred_hist_length = st.slider("Credit History Length :", min_value=0.0, max_value=50.0)
    credit_score = st.slider("Credit Score :", min_value=300, max_value=900)
    
    
    
    
with col3:
    person_education = st.selectbox("Education:", options = df["person_education"].unique())
    person_home_ownership = st.selectbox("Home Ownership :", options = df["person_home_ownership"].unique())
    loan_intent = st.selectbox("Loan Intent :", options = df["loan_intent"].unique())
    previous_loan_defaults_on_file= st.selectbox("loan intent:" , options = df["previous_loan_defaults_on_file"].unique())
    
    
# prepare new data
new_data = {'person_age': person_age, 'person_gender': person_gender, 'person_education': person_education, 'person_income': person_income,
       'person_emp_exp': person_emp_exp, 'person_home_ownership': person_home_ownership, 'loan_amnt': loan_amnt, 'loan_intent': loan_intent,
       'loan_int_rate': loan_int_rate, 'loan_percent_income': loan_percent_income, 'cb_person_cred_hist_length': cb_person_cred_hist_length,
       'credit_score': credit_score, 'previous_loan_defaults_on_file': previous_loan_defaults_on_file}

new_data = pd.DataFrame(new_data, index=[0])

with col4:
    if st.button("Predict"):
        prediction = best_pipeline.predict(new_data)[0]
        if prediction == 0:
            st.subheader("Loan Approved.")
        else:
            st.subheader("Loan Rejected.")