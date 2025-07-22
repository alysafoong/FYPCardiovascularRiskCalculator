import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("logreg_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the correct order of features
FEATURE_COLUMNS = [
    "Sex", "GeneralHealth", "PhysicalHealthDays", "MentalHealthDays", "LastCheckupTime",
    "PhysicalActivities", "SleepHours", "RemovedTeeth", "HadAngina", "HadStroke",
    "HadAsthma", "HadSkinCancer", "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease",
    "HadArthritis", "HadDiabetes", "DeafOrHardOfHearing", "DifficultyConcentrating",
    "DifficultyWalking", "DifficultyDressingBathing", "SmokerStatus", "ChestScan",
    "RaceEthnicityCategory", "AgeCategory", "BMI", "AlcoholDrinkers"
]

# Mapping dictionaries (based on how you encoded your features)
sex_map = {"Female": 0, "Male": 1}
last_checkup_map = {
    "Within past year (anytime less than 12 months ago)": 0,
    "Within past 2 years (1 year but less than 2 years ago)": 1,
    "Within past 5 years (2 years but less than 5 years ago)": 2,
    "5 or more years ago": 3
}
yes_no_map = {"No": 0, "Yes": 1}
removed_teeth_map = {
    "None of them": 0,
    "1 to 5": 1,
    "6 or more, but not all": 2,
    "All": 3
}
diabetes_map = {
    "No": 0,
    "Yes (other than pregnancy)": 1,
    "Yes, but only during pregnancy (female)": 2,
    "No, pre-diabetes or borderline diabetes": 3
}
smoker_map = {
    "Never smoked": 0,
    "Former smoker": 1,
    "Current smoker - now smokes some days": 2,
    "Current smoker - now smokes every day": 3
}
race_map = {
    "White only, Non-Hispanic": 0,
    "Black only, Non-Hispanic": 1,
    "Other race only, Non-Hispanic": 2,
    "Multiracial, Non-Hispanic": 3,
    "Hispanic": 4
}
age_map = {
    "Age 18 to 24": 0, "Age 25 to 29": 1, "Age 30 to 34": 2, "Age 35 to 39": 3,
    "Age 40 to 44": 4, "Age 45 to 49": 5, "Age 50 to 54": 6, "Age 55 to 59": 7,
    "Age 60 to 64": 8, "Age 65 to 69": 9, "Age 70 to 74": 10,
    "Age 75 to 79": 11, "Age 80 or older": 12
}

# Streamlit app
st.title("Cardiovascular Risk Calculator")

# Input form
with st.form("input_form"):
    st.markdown("### Gender")
    Sex = st.selectbox("", list(sex_map.keys()))
    
    st.markdown("### General Health")
    st.markdown("_Would you say that your health in general is: Poor (0), Fair (1), Good (2), Very Good (3), or Excellent (4)?_")
    GeneralHealth = st.slider("", 0, 4, 2)

    st.markdown("### Physical Health")
    st.markdown("_Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health **not good**?_")
    PhysicalHealthDays = st.slider("", 0, 30, 0)

    st.markdown("### Mental Health")
    st.markdown("_Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health **not good**?_")
    MentalHealthDays = st.slider("", 0, 30, 0)
    
    st.markdown("### Last Medical Checkup Time")
    st.markdown("_About how long has it been since you last visited a doctor for a routine checkup?_")
    LastCheckupTime = st.selectbox("", list(last_checkup_map.keys()))
    
    st.markdown("### Physical Activities")
    st.markdown("_During the **past month, other than your regular job**, did you participate in any physical activities or exercises such as running, calisthenics, golf, gardening, or walking for exercise?_")
    PhysicalActivities = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### Sleep Hours")
    st.markdown("_On average, how many hours of sleep do you get in a 24-hour period?_")
    SleepHours = st.slider("", 0, 24, 7)
    
    st.markdown("### Number of Permanent Teeth Removed")
    st.markdown("_Not including teeth lost for injury or orthodontics, how many of your permanent teeth have been removed because of tooth decay or gum disease?_")
    RemovedTeeth = st.selectbox("", list(removed_teeth_map.keys()))
    
    st.markdown("### Angina or Coronary Heart Disease")
    st.markdown("_Ever told you had angina or coronary heart disease?_")
    HadAngina = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### Stroke")
    st.markdown("_Ever told you had a stroke?_")
    HadStroke = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### Asthma")
    st.markdown("_Ever told you had asthma?_")
    HadAsthma = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### Skin Cancer")
    st.markdown("_Ever told you had skin cancer that is not melanoma?_")
    HadSkinCancer = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### COPD")
    st.markdown("_Ever told you had COPD (chronic obstructive pulmonary disease), emphysema or chronic bronchitis?_")
    HadCOPD = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### Depressive Disorder")
    st.markdown("_Ever told you had a depressive disorder (including depression, major depression, dysthymia, or minor depression)?_")
    HadDepressiveDisorder = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### Kidney Disease")
    st.markdown("_Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?_")
    HadKidneyDisease = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### Arthritis")
    st.markdown("_Ever told you had some form of arthritis, rheumatiod arthritis, gout, lupus, or fibromyalgia?_")
    HadArthritis = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### Diabetes")
    st.markdown("_Ever told you had diabetes?_")
    HadDiabetes = st.selectbox("", list(diabetes_map.keys()))
    
    st.markdown("### Deaf or Hard of Hearing")
    st.markdown("_Are you deaf or do you have serious difficulty hearing?_")
    DeafOrHardOfHearing = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### Difficulty Concentrating or Remembering")
    st.markdown("_Because of a physical, mental, or emotional condition, do you have serious difficulty concentrating, remembering, or making decisions?_")
    DifficultyConcentrating = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### Difficulty Walking or Climbing Stairs")
    st.markdown("_Do you have serious difficulty walking or climbing stairs?_")
    DifficultyWalking = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### Difficulty Dressing or Bathing")
    st.markdown("_Do you have difficulty dressing or bathing?_")
    DifficultyDressingBathing = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### Smoking Status")
    SmokerStatus = st.selectbox("", list(smoker_map.keys()))
    
    st.markdown("### Chest Scan")
    st.markdown("_Have you ever had a CT or CAT scan of your chest area?_")
    ChestScan = st.selectbox("", list(yes_no_map.keys()))
    
    st.markdown("### Race or Ethnicity")
    RaceEthnicityCategory = st.selectbox("", list(race_map.keys()))
    
    st.markdown("### Age")
    AgeCategory = st.selectbox("", list(age_map.keys()))
    
    st.markdown("### BMI")
    BMI = st.slider("", 10.0, 50.0, 22.0)
    
    st.markdown("### Alcohol Drinking")
    st.markdown("_Have you had at least one drink of alcohol in the **past 30 days**?_")
    AlcoholDrinkers = st.selectbox("", list(yes_no_map.keys()))

    submitted = st.form_submit_button("Predict")

if submitted:
    # Encode the inputs
    input_data = [
        sex_map[Sex],
        GeneralHealth,
        PhysicalHealthDays,
        MentalHealthDays,
        last_checkup_map[LastCheckupTime],
        yes_no_map[PhysicalActivities],
        SleepHours,
        removed_teeth_map[RemovedTeeth],
        yes_no_map[HadAngina],
        yes_no_map[HadStroke],
        yes_no_map[HadAsthma],
        yes_no_map[HadSkinCancer],
        yes_no_map[HadCOPD],
        yes_no_map[HadDepressiveDisorder],
        yes_no_map[HadKidneyDisease],
        yes_no_map[HadArthritis],
        diabetes_map[HadDiabetes],
        yes_no_map[DeafOrHardOfHearing],
        yes_no_map[DifficultyConcentrating],
        yes_no_map[DifficultyWalking],
        yes_no_map[DifficultyDressingBathing],
        smoker_map[SmokerStatus],
        yes_no_map[ChestScan],
        race_map[RaceEthnicityCategory],
        age_map[AgeCategory],
        BMI,
        yes_no_map[AlcoholDrinkers]
    ]

    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Results")
    st.write("Risk of Cardiovascular Disease:", "Yes, High Cardiovascular Risk" if prediction == 1 else "No, Low Cardiovascular Risk")
    st.write(f"Predicted Probability: {prediction_proba:.2%}")
