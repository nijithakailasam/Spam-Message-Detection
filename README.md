# 📩 Spam Message Detection

A Machine Learning-based web application that detects whether an SMS message is **Spam** or **Not Spam**.

## 🚀 Project Overview

This project uses **Natural Language Processing (NLP)** and Machine Learning to classify SMS messages into two categories: **Spam** and **Ham (Not Spam)**.

The text messages are converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**, and the **Multinomial Naive Bayes** algorithm is used to predict the message category.

The trained model and TF-IDF vectorizer are saved using Pickle and integrated into a **Streamlit web application**, where users can enter a message and instantly check whether it is spam or not.

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Multinomial Naive Bayes
* Streamlit
* Pickle

## 🔄 Project Workflow

```text
SMS Dataset
     ↓
Data Preprocessing
     ↓
Train/Test Split (80/20)
     ↓
TF-IDF Feature Extraction
     ↓
Multinomial Naive Bayes
     ↓
Model Training
     ↓
Save Model & Vectorizer
     ↓
Streamlit Web Application
     ↓
Enter New Message
     ↓
Spam / Not Spam Prediction
```

## ✨ Features

* Detects spam and legitimate SMS messages
* Uses TF-IDF for text feature extraction
* Uses Multinomial Naive Bayes for classification
* Provides instant predictions through a web interface
* Simple and user-friendly Streamlit application
* Trained model can be reused without retraining

## 📂 Project Structure

```text
Spam-Message-Detection/
│
├── dataset/
│   └── archive/
│       ├── SMSSpamCollection.csv
│       └── SpamCollectionSMS.txt
│
├── model/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
├── train.py
└── app.py
```

## 🎯 Objective

The main objective of this project is to build a simple and practical system that can automatically identify unwanted spam messages and help users recognize potentially suspicious SMS content.

## 🔮 Future Improvements

* Improve accuracy using larger and more diverse datasets
* Support multiple languages
* Add probability/confidence scores
* Explore advanced NLP and deep learning models
* Deploy the application online
