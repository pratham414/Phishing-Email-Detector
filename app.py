import streamlit as st
import pickle

# Load the trained models
with open("svm_model_poly.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Set page config for a modern UI
st.set_page_config(
    page_title="Email Type Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar with instructions
st.sidebar.title("🔍 How to Use")
st.sidebar.write("""
1️⃣ **Enter** the email subject & text  
2️⃣ **Click 'Predict'** to analyze the type  
3️⃣ **See the result** with a confidence score  
""")
st.sidebar.info("🚀 Powered by **SVM + TF-IDF**")

# Main UI
st.title("📩 Email Type Detection")
st.subheader("Classify emails as Fraud, Phishing, False Positives, or Spam!")

# Input fields
subject = st.text_input("📌 Email Subject:")
email_text = st.text_area("💬 Email Content:")

# Prediction button
if st.button("🔮 Predict Email Type"):
    if subject.strip() == "" or email_text.strip() == "":
        st.warning("⚠️ Please enter both subject and email content.")
    else:
        # Combine subject and text
        combined_text = subject + " " + email_text

        # Transform input text using the loaded TF-IDF vectorizer
        X_new = vectorizer.transform([combined_text])

        # Predict email type using the SVM model
        prediction = svm_model.predict(X_new)[0]

        # Display result with styled format
        st.success(f"✅ **Predicted Email Type:** {prediction}")

        # Footer
        st.write("🔹 _AI-powered email classification system_")
