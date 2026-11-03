
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import date,time,datetime

st.set_page_config(page_title="Premium Loan App",layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR/ "Smartpremiumloan.pkl"

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

final_pipeline = load_pipeline()

st.title("Premium Loan Amount Prediction App")
st.write("Enter the details for the prediction of the loan amount")
st.divider()

st.header("Customer Details")
col1,col2 = st.columns(2)
with col1:
    Age = st.number_input("Age",min_value=18,max_value=100,value=35,step=1)
    Gender = st.selectbox("Gender",["Female","Male"])
    Annual_Income = st.number_input("Annual Income",min_value=1.0,max_value=150000.0,value=80000.0,step=1000.0)
    Marital_Status = st.selectbox("Marital Status",["Married","Divorced","Single"])
with col2:
    Number_of_Dependents = st.number_input("Number of Dependents",min_value=0.0,max_value=4.0,value=2.8,step=0.1)
    Education_Level = st.selectbox("Education Level",["Bachelor's","Master's","High School","PhD"])
    Occupation = st.selectbox("Occupation",["Self-Employed","Employed","Unemployed"])
    Location = st.selectbox("Location",["Urban","Rural","Suburban"])

st.header("Health & Lifestyles")
col1,col2 = st.columns(2)
with col1:
    Health_Score = st.number_input("Health Score",min_value=2.0,max_value=59.0,value=30.2,step=1.0)
    Smoking_Status = st.selectbox("Smoking Status",["No","Yes"])
    Exercise_Frequency = st.selectbox("Exercise Frequency",["Weekly","Monthly","Daily","Rarely"])
with col2:
    Vehicle_Age = st.number_input("Vehicle Age",min_value=0.0,max_value=19.0,value=8.0,step=1.0)
    Property_Type = st.selectbox("Property Type",["House","Apartment","Condo"])

st.header("Insurance Information")
col1,col2 = st.columns(2)
with col1:
    Policy_Type = st.selectbox("Policy Type",["Premium","Comprehensive","Basic"])
    Previous_Claims = st.number_input("Previous Claims",min_value=0.0,max_value=9.0,value=2.5,step=0.1)
    Credit_Score = st.number_input("Credit Score",min_value=300.0,max_value=850.0,value=800.0,step=10.0)
with col2:
    Insurance_Duration = st.number_input("Insurance Duration",min_value=1.0,max_value=9.0,value=5.0,step=0.1)
    Customer_Feedback = st.selectbox("Customer Feedback",["Poor","Average","Good"])

st.header("Policy Start Date & Time")
col1,col2 = st.columns(2)
with col1:
    policy_date = st.date_input("Policy Start Date",value=date.today())
with col2:
    policy_time = st.time_input("Policy Start Time",value=time(12,0,0),step=60)

if st.button("Predict Premium",type="primary",use_container_width=True):
    try:
        policy_datetime = datetime.combine(policy_date,policy_time)
        Year = policy_datetime.year
        Month = policy_datetime.month
        Day = policy_datetime.day
        DayofYear = policy_datetime.timetuple().tm_yday
        Hour = policy_datetime.hour
        Minute = policy_datetime.minute

        customer_data = pd.DataFrame({
           "Age":[int(Age)],
            "Gender":[Gender],
            "Annual Income":[float(Annual_Income)],
            "Marital Status":[Marital_Status],
            "Number of Dependents":[float(Number_of_Dependents)],
            "Education Level":[Education_Level],
            "Occupation":[Occupation],
            "Location":[Location],
            "Health Score":[float(Health_Score)],
            "Smoking Status":[Smoking_Status],
            "Exercise Frequency":[Exercise_Frequency],
            "Vehicle Age":[float(Vehicle_Age)],
            "Property Type":[Property_Type],
            "Policy Type":[Policy_Type],
            "Previous Claims":[float(Previous_Claims)],
            "Credit Score":[float(Credit_Score)],
            "Insurance Duration":[float(Insurance_Duration)],
            "Customer Feedback":[Customer_Feedback],
            "Year":[Year],
            "Month":[Month],
            "Day":[Day],
            "DayofYear":[DayofYear],
            "Hour":Hour,
            "Minute":Minute
        })

        prediction = final_pipeline.predict(customer_data)
        premium = max(0,float(prediction[0]))

        st.success("Premium Loan Amount App is Predicting the Amount Successfully")
        st.subheader(f"Estimated Premium Amount: Rs: {premium:,.2f}")

    except Exception as e:
        st.error("An error occurred while generating the amount")
        st.exception(e)
