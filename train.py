import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv(
    "dataset/archive/SpamCollectionSMS.txt",
    sep="\t",
    names=["label", "message"]
)


# ==========================================
# 2. CHECK DATASET
# ==========================================

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(data.head())

print("\nColumns:")
print(data.columns)

print("\nDataset shape:")
print(data.shape)


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")
print(data.isnull().sum())


# Remove missing values
data = data.dropna(subset=["label", "message"])


# ==========================================
# 4. CHECK SPAM / HAM COUNT
# ==========================================

print("\nMessage count:")
print(data["label"].value_counts())


# ==========================================
# 5. INPUT AND OUTPUT
# ==========================================

X = data["message"]
y = data["label"]


# ==========================================
# 6. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))


# ==========================================
# 7. TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF conversion completed!")


# ==========================================
# 8. CREATE AI MODEL
# ==========================================

model = MultinomialNB()


# ==========================================
# 9. TRAIN MODEL
# ==========================================

model.fit(X_train_tfidf, y_train)

print("AI model training completed!")


# ==========================================
# 10. PREDICTION
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 11. ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(f"Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 13. CREATE MODEL FOLDER
# ==========================================

os.makedirs("model", exist_ok=True)


# ==========================================
# 14. SAVE AI MODEL
# ==========================================

with open("model/spam_model.pkl", "wb") as file:
    pickle.dump(model, file)


# ==========================================
# 15. SAVE TF-IDF VECTORIZER
# ==========================================

with open("model/vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY")
print("==========================================")

print("AI Model   : model/spam_model.pkl")
print("Vectorizer : model/vectorizer.pkl")


# ==========================================
# 16. TEST SAMPLE MESSAGES
# ==========================================

sample_messages = [
    "Congratulations! You won a free prize. Click now!",
    "Hey, are you coming to college tomorrow?",
    "URGENT! You have won 10000 dollars. Claim now!",
    "Can you call me when you are free?"
]


# Convert messages into TF-IDF
sample_tfidf = vectorizer.transform(sample_messages)


# Predict
predictions = model.predict(sample_tfidf)


# ==========================================
# 17. DISPLAY PREDICTIONS
# ==========================================

print("\n==========================================")
print("SAMPLE PREDICTIONS")
print("==========================================")


for message, prediction in zip(sample_messages, predictions):

    if prediction == "spam":
        result = "SPAM"
    else:
        result = "NOT SPAM"

    print("\nMessage:", message)
    print("Prediction:", result)


print("\n==========================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("==========================================")