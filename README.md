# 📟 Telecom Churn Prediction and Analysis Pipeline
Provides an app to predict Churn for telecommunication companies. Utilises Machine Learning to identify key aspects of Churn, and displays predictions and insights interactively.

## 📦 Usage

Refer to Deployed [Application](https://telco-churn-analysis-qkqq9n3i8s8ifr5qfurj2w.streamlit.app/) for testing the pipeline.

## 💡 Project Structure
```
Telco-Churn-Analysis
│   README.md
│   requirements.txt
│   streamlit_app.py
│
├───dashboard
│       analysis.pbix
│
├───data
│       processed_telco_churn.csv
│       Telco_customer_churn.xlsx
│
├───models
│       model_features.pkl
│       scaler.joblib
│       xgb_scaled_pos_model.pkl
│
├───notebooks
│       eda_processing.ipynb
│       explainability.ipynb
│       modelling_evaluation.ipynb
│
└───results
        feature_importance.csv
        tenure.csv
        top_10.ipynb
        top_10_features.csv
        top_10_features_telco_churn.csv

```
## ⚙️ Environment Setup
1. We recommend using Anaconda for environment management.

2. Create the environment using the provided environment.yml, and activate it:
```
conda env create -f environment.yml
conda activate churnPrediction
```

## 📚 Guide for pipeline utilised and Working Explaination
### Dataset Used and preprocessing
- Download data set from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), save as `data/Telco_customer_churn.csv`.
- Navigate to `notebooks/eda_processing.ipynb`.
    - Renamed columns to remove whitespaces, and dropped NA values.
    - Utilised `seaborn` and `matplotlib` for observing trends via charts and heat maps (correlation matrix).
    - Applied binary encoding for boolean value columns.
    - Applied one-hot encoding for categorical features, and dropping the first category.
    - Bucketized tenure.
    - Standardized numeric features utilising `StandardScaler`.
    - Assigned Unique IDs (uuid) to each row.
- Save as `data/processed_telco_churn.csv`.

### Model Training and Evaluation
- Navigate to `notebooks/modelling_evaluation.ipynb`.
    - Assigned target label as `churn_label`, and utilised a 80-20 Train-Test split.
    - Utilised `XGBoost` for training model.
        - Sequentially improves decision trees by employing gradient boosting.
        - Primarily chosen for tracking feature importance.
    - Evaluated performance using Accuracy, F1 and ROC-AUC scores.
        - Accuracy
            - Measures overall correctness.
            - Not reliable for imbalanced datasets.
        - F1 Score
            - Tracks false positives and false negatives.
            - Reliable tracking method.
        - ROC-AUC Score
            - Tracks True positives and False positives.
            - Especially useful for Churn scenario, as cost of incorrectly predicting churn is significantly lower than incorrectly predicting no Churn.
    - Performance Improvement Methods, to handle class imbalance
        - SMOTE:
            - Generates synthetic samples of minority class.
        - Scale_POS_weight:
            - Feature of XGBoost
            - Treats minority class as of having greater importance.
    - Utilised scale_pos_weight over SMOTE to improve performance.
    - Used Optuna for finding appropriate hyperparamter values of XGBoost.
    - Observed marked improve in performance via tracking
        - Confusion matrix state
        - ROC AUC score
    - Saved model as a pickle file to `models/xgb_scaled_pos_model.pkl`.

### Explaination and analysis using SHAP
- Utilised SHAP graphs to observe features with highest impact on churn.
- Saved the top 10 features to `results/top_10_features.csv`

### Pipeline result and prototyping testing
- Deployed app encapsulating previously mentioned pipeline via `streamlit`.
- Utilised PowerBI to create an interactable dashboard.
