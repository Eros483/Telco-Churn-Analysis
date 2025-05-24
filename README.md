# Telco Churn Analysis

This is a Streamlit-based web application for analyzing customer churn data. The app loads pre-trained models and displays predictions and data insights interactively.

🔗 **Live App**: [Telco Churn Analysis](https://telco-churn-analysis-qkqq9n3i8s8ifr5qfurj2w.streamlit.app/)

---

## 📁 Project Structure

root/
├── app/
│ ├── streamlit_app.py # Main Streamlit application
│ └── loaders.py # Utility functions for loading models and data
├── data/
│ └── your_csv_file.csv # CSV dataset
├── models/
│ └── your_model_name.pkl # Serialized machine learning model(s)

yaml
Copy
Edit

---

## 🚀 How to Run Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
Install requirements

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app/streamlit_app.py
📦 Dependencies
streamlit

pandas

scikit-learn (or equivalent used to create the .pkl model)

pickle (standard library)

Install via:

bash
Copy
Edit
pip install streamlit pandas scikit-learn
📚 Files Explained
app/loaders.py
Contains reusable functions:

load_data() → Loads CSV from /data

load_model(model_name) → Loads a .pkl file from /models

Handles paths cleanly using os.path.

app/streamlit_app.py
Loads model and dataset using loaders.py

Displays head of the data

Indicates successful model loading

🔮 Future Enhancements
Dynamically list and select from available models in the /models directory

Add prediction interface for user input

Visualize feature importances

Improve deployment performance via model caching

📡 Deployment
This app is deployed on Streamlit Cloud.

👉 Live Demo: https://telco-churn-analysis-qkqq9n3i8s8ifr5qfurj2w.streamlit.app/
