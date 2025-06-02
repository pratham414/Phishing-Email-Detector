import streamlit as st
import pickle
import numpy as np

# Load the trained models
with open("svm_model_poly.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Optional: Define class labels
label_map = {
    0: "False Positive",
    1: "Spam",
    2: "Fraud",
    3: "Phishing"
}

# Set page config
st.set_page_config(
    page_title="Email Type Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar instructions
st.sidebar.title("🔍 How to Use")
st.sidebar.write("""
1️⃣ **Enter** the email subject & content  
2️⃣ **Click 'Predict'** to analyze the email type  
3️⃣ **See the result** below  
""")
st.sidebar.info("🚀 Powered by **SVM + TF-IDF**")

# Main UI
st.title("📩 Email Type Detection")
st.subheader("Classify emails as Fraud, Phishing, False Positives, or Spam")

# Input fields
subject = st.text_input("📌 Email Subject:")
email_text = st.text_area("💬 Email Content:")

# Predict button
if st.button("🔮 Predict Email Type"):
    if subject.strip() == "" or email_text.strip() == "":
        st.warning("⚠️ Please enter both the subject and content of the email.")
    else:
        combined_text = subject + " " + email_text
        X_new = vectorizer.transform([combined_text])
        prediction = svm_model.predict(X_new)[0]

        # Display result
        predicted_label = label_map.get(prediction, "Unknown")
        st.success(f"✅ **Predicted Email Type:** {predicted_label}")

        # If you want to show probabilities (if supported)
        if hasattr(svm_model, "decision_function"):
            score = np.max(svm_model.decision_function(X_new))
            st.write(f"📈 Confidence score: `{score:.2f}`")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit, SVM & TF-IDF")
