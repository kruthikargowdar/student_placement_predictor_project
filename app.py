import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from streamlit_lottie import st_lottie

# --- Setup ---
st.set_page_config(page_title="Student Placement Predictor", layout="centered")


def local_css(file_name):
    with open(file_name, encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")
# Inject floating emojis
emojis = ["🌟", "✨", "💫", "🌈"]
floating_emojis = "".join(
    f'<div class="floating-emoji" style="left:{i*20}%; top:{(i%3)*20}%">{emoji}</div>'
    for i, emoji in enumerate(emojis)
)
st.markdown(f"<div>{floating_emojis}</div>", unsafe_allow_html=True)

# Floating sparkles
st.markdown("""
<div class="floating" style="top:50px; left:30px;"></div>
<div class="floating" style="top:100px; left:150px;"></div>
<div class="floating" style="top:200px; left:90px;"></div>
""", unsafe_allow_html=True)


# --- Load Assets ---
def load_lottiefile(filepath):
    with open(filepath, "r") as f:
        return json.load(f)
    
lottie_success = load_lottiefile("assets/success.json")

# --- Load Model & Metadata ---
with open("placement_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

feature_columns = metadata["features"]
label_encoders = metadata["label_encoders"]

# --- UI ---
st.markdown("""
<h1 class="animated-header"> Student Placement Prediction</h1>
""", unsafe_allow_html=True)


st.subheader("📝 Enter Student Details")

user_input = {}
for feature in feature_columns:
    if feature in label_encoders:
        options = label_encoders[feature].classes_
        user_input[feature] = st.selectbox(feature, options)
    else:
        user_input[feature] = st.number_input(feature, step=0.1)

if st.button("🔮 Predict Placement"):
    with st.spinner("Analyzing..."):
        

        input_df = pd.DataFrame([user_input])
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]

        # Output result
        if prediction == 1:
            st.success("🎉 The student is likely to be placed!")
            st_lottie(lottie_success, speed=1, height=200)
        else:
            st.error("❌ The student is not likely to be placed.")

        # SHAP Explainability
        st.subheader("📊 Explanation of the Prediction")

        raw_df = pd.read_csv("Placement_Data_Full_Class.csv")
        X_train = raw_df[feature_columns].copy()
        for col in X_train.columns:
            if col in label_encoders:
                X_train[col] = label_encoders[col].transform(X_train[col])

        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(input_df)

        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)
        
        # 📊 Model Performance Metrics
with st.expander("📈 Model Evaluation Metrics"):
    try:
        raw_df = pd.read_csv("Placement_Data_Full_Class.csv")
        X = raw_df[feature_columns].copy()
        y = raw_df["status"].apply(lambda x: 1 if x == "Placed" else 0)
        for col in X.columns:
            if col in label_encoders:
                X[col] = label_encoders[col].transform(X[col])
        y_pred = model.predict(X)
        acc = (y == y_pred).mean()

        st.metric("Model Accuracy", f"{acc*100:.2f}%")
        st.progress(acc)

        col1, col2 = st.columns(2)
        with col1:
            st.write("🔁 Confusion Matrix")
            st.write(pd.crosstab(y, y_pred, rownames=["Actual"], colnames=["Predicted"]))
        with col2:
            st.write("📑 Classification Report")
            st.text(
                f"""Precision: {precision_score(y, y_pred):.2f}
                Recall:    {recall_score(y, y_pred):.2f}
                F1 Score:  {f1_score(y, y_pred):.2f}"""
                )

            
    except Exception as e:
        st.warning(f"Couldn't generate evaluation metrics: {e}")

st.markdown("<br><br>", unsafe_allow_html=True)
