# Telco Churn Analysis

This is a Streamlit-based web application for analyzing customer churn data. The app loads pre-trained models and displays predictions and data insights interactively.
The goal was to use machine learning to predict which customers are likely to cancel their subscription, and identify the key reasons behind churn.

🔗 **Live App**: [Telco Churn Analysis](https://telco-churn-analysis-qkqq9n3i8s8ifr5qfurj2w.streamlit.app/)

---
### Features
- Dataset taken from kaggle for training model with data on existing churn and other relevant information
- Exploratory Analysis and preprocessing on the dataset to encode categorical features, and normalize numerical ones, while sampling most relevant features
- Training XGBoost model to classify churn
- Integrated Optuna to for hyperparameter tuning to improve scores of the trained model
- Evaluated model using accuracy score, F1-Score and ROC-AUC score
- used SHAP to find which features most influence churn
- Built a powerBI dashboard showing visualising relevant factors
- Created a basic streamlit app with churn prediction capabilities, provided data.
---

##  How to Run Locally
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   pip install -r requirements.txt
```

### Dependencies
-pandas

-numpy

-matplotlib

-seaborn

-scikit-learn

-shap

-xgboost

-streamlit

-optuna

-imblearn

-joblib

-ipython
