import streamlit as st
import google.generativeai as genai
import time

# --- CONFIGURATION ---
# Set your Gemini API Key here (or use an environment variable)
API_KEY = "" 

# Configure Gemini API
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.warning("Please provide a Gemini API Key in the code to enable the chatbot.")

# --- PAGE CONFIG ---
st.set_page_config(page_title="PulseGuard AI - BP System", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stChatFloatingInputContainer { bottom: 20px; }
    .report-card { 
        background-color: white; 
        padding: 2rem; 
        border-radius: 15px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🏥 PulseGuard AI")
    app_mode = st.radio("Navigate", ["Chat Assistant", "System Documentation", "Sample ML Code"])
    st.info("This system explains and predicts risks for Hypertension and Heart Disease.")

# --- TAB 1: CHAT ASSISTANT ---
if app_mode == "Chat Assistant":
    st.header("💬 PulseGuard Chatbot")
    st.caption("Ask me anything about Blood Pressure Prediction, ML Models, or Preprocessing.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ex: What are the best metrics for BP prediction?"):
        
        with st.chat_message("user"):
            st.markdown(prompt)

      
            if not API_KEY:
                response_text = "API Key missing. Please check app.py."
            else:
                try:
                    # Provide context to Gemini for project-specific answers
                    context = "You are a specialist in Medical ML. The user is asking about a Blood Pressure/Heart Disease Prediction system."
                    full_prompt = f"{context}\nUser Query: {prompt}"
                    response = model.generate_content(full_prompt)
                    response_text = response.text
                except Exception as e:
                    response_text = f"Error: {str(e)}"
            
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- TAB 2: SYSTEM DOCUMENTATION ---
elif app_mode == "System Documentation":
    st.title("📑 System Design & Explanation")
    
    with st.container():
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.subheader("1. Problem Statement")
        st.write("""
        Hypertension is a 'silent killer.' Traditional diagnosis often relies on reactive clinical visits. 
        The goal is to develop a **Predictive Health System** that utilizes routine clinical features 
        (age, cholesterol, heart rate) to identify high-risk individuals before a crisis occurs.
        """)
        
        st.subheader("2. Dataset Description")
        st.write("""
        The system typically uses the **UCI Heart Disease Dataset**. Features include:
        - **Age / Sex**: Demographic factors.
        - **Chest Pain (cp)**: Categorical (4 types).
        - **Resting BP (trestbps)**: mmHg.
        - **Cholesterol (chol)**: mg/dl.
        - **Fasting Blood Sugar (fbs)**: Binary (> 120 mg/dl).
        - **Maximum Heart Rate (thalach)**: BPM.
        """)

        st.subheader("3. Preprocessing Steps")
        st.markdown("""
        - **Handling Missing Values**: Imputation using mean (for BP) or mode (for categorical).
        - **Encoding**: One-Hot Encoding for 'Chest Pain' and 'Thal'.
        - **Scaling**: StandardScaler or RobustScaler for continuous variables like Cholesterol and Heart Rate.
        - **Outlier Detection**: Using Z-score to remove extreme BP readings.
        """)

        st.subheader("4. Model Selection & Training")
        st.write("""
        - **Baseline**: Logistic Regression (high interpretability).
        - **Champion Model**: Random Forest or XGBoost (handles non-linear medical relationships).
        - **Cross-Validation**: K-Fold (K=5) to ensure stability across different patient groups.
        """)

        st.subheader("5. Evaluation Metrics")
        st.markdown("""
        - **Sensitivity (Recall)**: Critical in medical apps to minimize False Negatives.
        - **Precision**: To reduce False Alarms.
        - **AUC-ROC**: To measure the model's ability to distinguish between healthy and at-risk patients.
        """)
        
        st.subheader("6. Deployment Strategy (Flask)")
        st.code("""
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('bp_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([list(data.values())])
    return jsonify({'risk_level': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
        """, language='python')

        st.subheader("7. Possible Improvements")
        st.write("""
        - **Temporal Data**: Incorporating longitudinal patient history (tracking BP over months).
        - **Explainability**: Using SHAP or LIME to tell the doctor *why* a patient is flagged.
        - **IoT Integration**: Real-time data from wearable pulse guards.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: SAMPLE ML CODE ---
elif app_mode == "Sample ML Code":
    st.header("🛠️ Implementation Blueprint")
    st.write("Below is a conceptual script for training the system.")
    
    code_snippet = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load Data
df = pd.read_csv('hypertension_data.csv')

# 2. Preprocess
X = df.drop('target', axis=1)
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# 4. Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
    """
    st.code(code_snippet, language='python')
