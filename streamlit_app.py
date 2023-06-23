import streamlit as st
import pandas as pd
import joblib

st.title("10 Years Heart Disease prediction")

col1, col2, col3 = st.columns(3)
gender = col1.selectbox("Enter your gender", ["Male", "Female"])
age = col2.number_input("Enter your age")
education = col3.selectbox("Highest academic qualification", ["High School Diploma", "Undergraduate Degree", "Postgraduate Degree", "PhD"])

isSmoker = col1.selectbox("Are you currently a Smoker?", ["Yes", "No"])
yearsSmoking = col2.number_input("Number of Daily Cigarettes")
BPMeds = col3.selectbox("Are you currently on BP medication?", ["Yes", "No"])

stroke = col1.selectbox("Have you ever experienced a stroke?", ["Yes", "No"])
hyp = col2.selectbox("Do you have hypertension?", ["Yes", "No"])
diabetes = col3.selectbox("Do you have diabetes?", ["Yes", "No"])

chol = col1.number_input("Enter your cholesterol level")
sys_bp = col2.number_input("Enter your systolic blood pressure")
dia_bp = col3.number_input("Enter your diastolic blood pressure")

bmi = col1.number_input("Enter your BMI")
heart_rate = col2.number_input("Enter your resting heart rate")
glucose = col3.number_input("Enter your glucose level")

predict_button = st.button("Predict")

df_pred = pd.DataFrame(
    [[gender, age, education, isSmoker, yearsSmoking, BPMeds, stroke, hyp, diabetes, chol, sys_bp, dia_bp, bmi,
      heart_rate, glucose]],
    columns=['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
             'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
)

df_pred['male'] = df_pred['male'].apply(lambda x: 1 if x == 'Male' else 0)
df_pred['currentSmoker'] = df_pred["currentSmoker"].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['prevalentStroke'] = df_pred["prevalentStroke"].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['prevalentHyp'] = df_pred["prevalentHyp"].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['diabetes'] = df_pred["diabetes"].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['BPMeds'] = df_pred["BPMeds"].apply(lambda x: 1 if x == 'Yes' else 0)

mapping = {
    "High School Diploma": 0,
    "Undergraduate Degree": 1,
    "Postgraduate Degree": 2,
    "PhD": 3
}
df_pred['education'] = df_pred['education'].map(mapping)


model = joblib.load('rf_model.pkl')
prediction = model.predict(df_pred)

if predict_button:
    if(prediction[0]==0):
        st.write("You will not develop a heart disease in 10 years likely")
    else:
        st.write("You are likely to develop a heart disease within the next 10 years")
