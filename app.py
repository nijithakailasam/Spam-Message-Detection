import streamlit as st
import pickle

# Load trained model and vectorizer
with open("model/spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("model/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Page configuration
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered"
)

# Title
st.title("📩 Spam Message Detection")
st.write("Enter a message below to check whether it is Spam or Not Spam.")


# Message input
message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You won a free prize..."
)


# Prediction button
if st.button("🔍 Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        # Convert message into TF-IDF features
        message_vector = vectorizer.transform([message])

        # Make prediction
        prediction = model.predict(message_vector)[0]

        # Display result
        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE")
            st.write("This message looks like spam.")

        else:
            st.success("✅ NOT SPAM")
            st.write("This message looks safe.")