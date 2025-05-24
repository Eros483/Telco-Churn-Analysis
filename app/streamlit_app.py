import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
from pathlib import Path

#paths
model_path = Path.cwd().parent/"models"/"xgb_scaled_pos_model.pkl"
data_path=Path.cwd().parent/"data"/"processed_telco_churn.csv"
scaler_path=Path.cwd().parent/"models"/"scaler.joblib"
model_features_path=Path.cwd().parent/"models"/"model_features.pkl"

#model
model=joblib.load(model_path)
model.set_params(device='cpu')

#scaler
scaler=joblib.load(scaler_path)

#data
df=pd.read_csv(data_path)
df=df.drop(columns=["churn_label"], errors="ignore")

#features
features=df.columns.tolist()
model_features=joblib.load(model_features_path)

#shap explainer
explainer=shap.TreeExplainer(model)

st.set_page_config(page_title="Churn Predicter", layout="wide")
st.title("Telco Customer Churn Predictor")

#Query mode, upload or manual
input_mode=st.radio("Choose query type", ["Upload CSV (for multiple customer data)", "Manual Entry (for single customer data)"])
uploaded_df = pd.DataFrame()

#for uploaded CSV
if input_mode=="Upload CSV (for multiple customer data)":
    st.markdown("Required CSV Format")
    st.info("Your CSV file should have the following columns:")
    st.code(", ".join(features), language="csv")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        uploaded_df=pd.read_csv(uploaded_file)
        if set(features).issubset(uploaded_df.columns):
            st.success("Valid Input data file  received")
        else:
            st.error("Invalid Input data file. Please ensure the CSV file has the required columns.")
            st.stop()

#for manual entry
else:
    st.subheader("Enter Customer Details")
    input_dict={}
    
    st.markdown("### Basic Information")
    col1, col2=st.columns(2)
    with col1:
        gender=st.selectbox("Gender", ["Male", "Female"])
        if gender=="Male":
            input_dict["Gender"]=1
        else:
            input_dict["Gender"]=0
        senior=st.selectbox("Are you a Senior Citizen", ["Yes", "No"])
        if senior=="Yes":
            input_dict["Senior Citizen"]=1
        else:
            input_dict["Senior Citizen"]=0

    with col2:
        partner=st.selectbox("Do you have a partner?", ["Yes", "No"])
        if partner=="Yes":
            input_dict["Partner"]=1
        else:
            input_dict["Partner"]=0

        Dependent=st.selectbox("Do you have dependents?", ["Yes", "No"])
        if Dependent=="Yes":
            input_dict["Dependents"]=1
        else:
            input_dict["Dependents"]=0

    st.markdown('### Account Duration and Charges')
    col3, col4=st.columns(2)
    with col3:
        input_dict["tenure"]=st.number_input("Tenure in months", min_value=0.0, step=1.0)
        input_dict["MonthlyCharges"]=st.number_input("Monthly Charges", min_value=0.0, step=0.1)
    with col4:
        input_dict["TotalCharges"]=st.number_input("Total Charges", min_value=0.1)
        tenure_bucket=st.selectbox("Tenure Bucket", ["none", "short", "medium", "long", "full time"])
        if tenure_bucket=="none":
            input_dict["tenure_bucket"]=0
        elif tenure_bucket=="short":
            input_dict["tenure_bucket"]=1
        elif tenure_bucket=="medium":
            input_dict["tenure_bucket"]=2
        elif tenure_bucket=="long":
            input_dict["tenure_bucket"]=3
        else:
            input_dict["tenure_bucket"]=4

    st.markdown('### Services signed up for')
    services=[
        "Phone Service", 
        "MultipleLines_Yes_True",
        "Internet Service_Fiber optic_True", 
        "OnlineSecurity_Yes_True",
        "OnlineBackup_Yes_True",
        "DeviceProtection_Yes_True",
        "TechSupport_Yes_True",
        "Streaming TV_Yes_True",
        "Streaming Movies_Yes_True",
    ]
    for i in range(0, len(services), 2):
        col_a, col_b = st.columns(2)
        with col_a:
            service=st.selectbox(f"{services[i]}", ["True", "False"])
            if service=="True":
                input_dict[services[i]]=1
            else:
                input_dict[services[i]]=0
        if i + 1 < len(services):
            with col_b:
                alt_service= st.selectbox(f"{services[i + 1]}", ["True", "False"])
                if alt_service=="True":
                    input_dict[services[i + 1]]=1
                else:
                    input_dict[services[i + 1]]=0

    if input_dict['MultipleLines_Yes_True']==1:
        input_dict['MultipleLines_Yes_False']=0
    else:
        input_dict['MultipleLines_Yes_False']=1

    if input_dict['Internet Service_Fiber optic_True']==1:
        input_dict['Internet Service_Fiber optic_False']=0
    else:
        input_dict['Internet Service_Fiber optic_False']=1

    if input_dict['OnlineSecurity_Yes_True']==1:
        input_dict['OnlineSecurity_Yes_False']=0
    else:
        input_dict['OnlineSecurity_Yes_False']=1

    if input_dict['OnlineBackup_Yes_True']==1:
        input_dict['OnlineBackup_Yes_False']=0
    else:
        input_dict['OnlineBackup_Yes_False']=1

    if input_dict['DeviceProtection_Yes_True']==1:
        input_dict['DeviceProtection_Yes_False']=0
    else:
        input_dict['DeviceProtection_Yes_False']=1

    if input_dict['TechSupport_Yes_True']==1:
        input_dict['TechSupport_Yes_False']=0
    else:
        input_dict['TechSupport_Yes_False']=1

    if input_dict['Streaming TV_Yes_True']==1:
        input_dict['Streaming TV_Yes_False']=0
    else:
        input_dict['Streaming TV_Yes_False']=1

    if input_dict['Streaming Movies_Yes_True']==1:
        input_dict['Streaming Movies_Yes_False']=0
    else:
        input_dict['Streaming Movies_Yes_False']=1

    st.markdown("### Contracts and Payment Info")
    col8, col9=st.columns(2)
    with col8:
        contract_length=st.selectbox("Length of Contract (years)", ["0", "1", "2"])
        if contract_length=="0":
            input_dict["Contract_One year_False"]=1
            input_dict["Contract_One year_True"]=0
            input_dict["Contract_Two year_False"]=1
            input_dict["Contract_Two year_True"]=0
        
        elif contract_length=="1":
            input_dict["Contract_One year_False"]=0
            input_dict["Contract_One year_True"]=1
            input_dict["Contract_Two year_False"]=1
            input_dict["Contract_Two year_True"]=0

        else:
            input_dict["Contract_One year_False"]=1
            input_dict["Contract_One year_True"]=0
            input_dict["Contract_Two year_False"]=0
            input_dict["Contract_Two year_True"]=1

    with col9:
        payment_method=st.selectbox("What is your payment method?", ["Credit Card", "Electronic Check", "Mailed check", "none"])
        if payment_method=="Credit Card":
            input_dict["Payment Method_Credit card (automatic)_False"]=0
            input_dict["Payment Method_Credit card (automatic)_True"]=1
            input_dict["Payment Method_Electronic check_False"]=1
            input_dict["Payment Method_Electronic check_True"]=0
            input_dict["Payment Method_Mailed check_False"]=1
            input_dict["Payment Method_Mailed check_True"]=0
        
        elif payment_method=="Electronic Check":
            input_dict["Payment Method_Credit card (automatic)_False"]=1
            input_dict["Payment Method_Credit card (automatic)_True"]=0
            input_dict["Payment Method_Electronic check_False"]=0
            input_dict["Payment Method_Electronic check_True"]=1
            input_dict["Payment Method_Mailed check_False"]=1
            input_dict["Payment Method_Mailed check_True"]=0
        
        elif payment_method=="Mailed Check":
            input_dict["Payment Method_Credit card (automatic)_False"]=1
            input_dict["Payment Method_Credit card (automatic)_True"]=0
            input_dict["Payment Method_Electronic check_False"]=1
            input_dict["Payment Method_Electronic check_True"]=0
            input_dict["Payment Method_Mailed check_False"]=0
            input_dict["Payment Method_Mailed check_True"]=1
        
        else:
            input_dict["Payment Method_Credit card (automatic)_False"]=1
            input_dict["Payment Method_Credit card (automatic)_True"]=0
            input_dict["Payment Method_Electronic check_False"]=1
            input_dict["Payment Method_Electronic check_True"]=0
            input_dict["Payment Method_Mailed check_False"]=1
            input_dict["Payment Method_Mailed check_True"]=0

    st.markdown("### Auxiliary Features")
    aux=[
        "Paperless Billing",
        "Multiple Lines_No phone service_True" ,
        "Multiple Lines_Yes_True",
        "Internet Service_No_True",
        "Online Security_No internet service_True",
        "Online Security_Yes_True",
        "Online Backup_No internet service_True",
        "Online Backup_Yes_True",
        "Device Protection_No internet service_True",
        "Device Protection_Yes_True",
        "Tech Support_No internet service_True",
        "Tech Support_Yes_True",
        "Streaming TV_No internet service_True",
        "Streaming Movies_No internet service_True"
    ]
    for feature in aux:
        input_dict[feature] = 0

    for i in range(0, len(aux), 3):
        col_d, col_e, col_f = st.columns(3)
        with col_d:
            auxa=st.selectbox(f"{aux[i]}", ["True", "False"], key=f"{i}")
            if auxa=="True":
                input_dict[aux[i]]=1
            else:
                input_dict[aux[i]]=0
        if i + 1 < len(aux):
            with col_e:
                auxb= st.selectbox(f"{aux[i + 1]}", ["True", "False"], key=f"{i+1}")
                if auxb=="True":
                    input_dict[aux[i + 1]]=1
                else:
                    input_dict[aux[i + 1]]=0
        if i + 2 < len(aux):
            with col_f:
                auxc= st.selectbox(f"{aux[i + 2]}", ["True", "False"], key=f"{i+2}")
                if auxc=="True":
                    input_dict[aux[i + 1]]=1
                else:
                    input_dict[aux[i + 1]]=0

    if input_dict.get('Streaming Movies_No internet service_True', 0)==1:
        input_dict['Streaming Movies_No internet service_False']=0
    else:
        input_dict['Streaming Movies_No internet service_False']=1

    if input_dict.get('Streaming TV_No internet service_True', 0)==1:
        input_dict['Streaming TV_No internet service_False']=0
    else:
        input_dict['Streaming TV_No internet service_False']=1
    
    if input_dict.get('Tech Support_Yes_True', 0)==1:
        input_dict['Tech Support_Yes_False']=0
    else:
        input_dict['Tech Support_Yes_False']=1

    if input_dict.get('Tech Support_No internet service_True', 0)==1:
        input_dict['Tech Support_No internet service_False']=0
    else:
        input_dict['Device Protection_Yes_False']=1

    if input_dict.get('Device Protection_Yes_True', 0)==1:
        input_dict['Device Protection_Yes_False']=0
    else:
        input_dict['Device Protection_Yes_False']=1

    if input_dict.get('Device Protection_No internet service_True', 0)==1:
        input_dict['Device Protection_No internet service_False']=0
    else:
        input_dict['Device Protection_No internet service_False']=1

    if input_dict.get('Online Backup_Yes_True', 0)==1:
        input_dict['Online Backup_Yes_False']=0
    else:
        input_dict['Online Backup_Yes_False']=1

    if input_dict.get('Online Backup_No internet service_True', 0)==1:
        input_dict['Online Backup_No internet service_False']=0
    else:
        input_dict['Online Backup_No internet service_False']=1

    if input_dict.get('Online Security_No internet service_True', 0)==1:
        input_dict['Online Security_No internet service_False']=0
    else:
        input_dict['Online Security_No internet service_False']=1

    if input_dict.get('Online Security_Yes_True', 0)==1:
        input_dict['Online Security_Yes_False']=0
    else:
        input_dict['Online Security_Yes_False']=1

    if input_dict.get('Internet Service_No_True', 0)==1:
        input_dict['Internet Service_No_False']=0
    else:
        input_dict['Internet Service_No_False']=1

    if input_dict.get('Multiple Lines_No phone service_True', 0)==1:
        input_dict['Multiple Lines_No phone service_False']=0
    else:
        input_dict['Multiple Lines_No phone service_False']=1

    if input_dict.get('Multiple Lines_Yes_True', 0)==1:
        input_dict['Multiple Lines_Yes_False']=0
    else:
        input_dict['Multiple Lines_Yes_False']=1

    uploaded_df=pd.DataFrame([input_dict])

    misc = [f for f in features if f not in input_dict]
    for i in range(0, len(misc), 2):
        col_e, col_f = st.columns(2)
        with col_e:
            input_dict[misc[i]] = st.selectbox(f"{misc[i]}", ["True", "False"])
        if i + 1 < len(misc):
            with col_f:
                input_dict[misc[i + 1]] = st.selectbox(f"{misc[i + 1]}", ["True", "False"])

#numerical data scaling to fit processed data distribution
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
if not uploaded_df.empty:
    existing_numeric_cols=[col for col in numerical_cols if col in uploaded_df.columns]
    uploaded_df[existing_numeric_cols]=scaler.transform(uploaded_df[existing_numeric_cols])

#prediction
st.subheader("Prediction Results")
if not uploaded_df.empty:
    try:
        for feature in model_features:
            if feature not in uploaded_df.columns:
                uploaded_df[feature] = 0

        uploaded_df_processed=uploaded_df[model_features]

        probs=model.predict_proba(uploaded_df_processed)[:, 1]
        labels=(probs>=0.5).astype(int)

        uploaded_df["Churn Probability"]=probs
        uploaded_df["Churn Prediction"]=labels
        uploaded_df["Churn Prediction"]=uploaded_df["Churn Prediction"].replace({1: "Churn", 0: "Not Churn"})
        st.dataframe(uploaded_df[["Churn Probability", "Churn Prediction"]])

        #SHAP explainability
        st.subheader("SHAP Explanation")

        shap_values=explainer.shap_values(uploaded_df_processed)
        shap.initjs()

        if isinstance(shap_values, list):
            shap_values=shap_values[0]
        else:
            shap_values=shap_values

        plt.figure(figsize=(5,2))
        shap.summary_plot(shap_values, uploaded_df[model_features], show=False)
        st.pyplot(plt)

    except ValueError as e:
        st.error("Error: Please upload csv")

            