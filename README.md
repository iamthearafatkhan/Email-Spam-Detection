# A Transformer-Based Approach for Email Spam Detection Using DistilBERT (Spam Detection App)
Streamlit app using DistilBERT loaded from Hugging Face Hub.
email spam classification system built using DistilBERT, TF-IDF + ML models, and deployed using Streamlit. Main model used DistilBERT which is transformer based model. The system classifies messages as Spam or Ham with high accuracy and provides confidence scores.

📂 Project Overview
-This project focuses on detecting spam emails using multiple approaches:
-✅ Traditional Machine Learning (TF-IDF + Naive Bayes, SVM)
-✅ Deep Learning (LSTM)
-✅ Transformer Model (DistilBERT — Main Model)
-✅ Fully deployed using Streamlit
-✅ Custom-trained DistilBERT model, uploaded to HuggingFace
-=> The goal is to improve spam detection accuracy and deploy a real-time working web application.


🧠 Features
-✔ DistilBERT-based spam classifier
-✔ Additional ML models for comparison
-✔ Real-time message prediction
-✔ File upload (.txt) prediction
-✔ Background image + light/dark theme
-✔ Confidence score + probability visualization
-✔ Deployed on Streamlit Cloud
-✔ HuggingFace model hosting


🛠️ Tech Stack
-✔ Main Model: 
-DistilBERT (HuggingFace Transformers)
-✔ Other Models:
-TF-IDF + Naive Bayes
-TF-IDF + SVM
-LSTM (Keras)
-✔ Frameworks
-Streamlit (UI + Deployment)
-PyTorch
-Transformers
-Scikit-learn
-Pandas, NumPy


📦 Model Loading
-The app loads your custom model directly from HuggingFace:
 - model_repo = "iamthearafatkhan/distilbert-spam2336"
 - tokenizer = AutoTokenizer.from_pretrained(model_repo)
 - model = AutoModelForSequenceClassification.from_pretrained(model_repo)
