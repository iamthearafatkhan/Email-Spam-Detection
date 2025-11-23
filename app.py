import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import base64
from pathlib import Path


# STREAMLIT PAGE CONFIG

st.set_page_config(
    page_title="Spam Detector | DistilBERT",
    page_icon="📧",
    layout="centered"
)





# BACKGROUND IMAGE

def set_bg(image_file):
    if Path(image_file).exists():
        with open(image_file, "rb") as f:
            img_data = f.read()
        b64_img = base64.b64encode(img_data).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/jpg;base64,{b64_img}");
                background-size: cover;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

set_bg("bgb2.jpg")


# LOAD MODEL

@st.cache_resource
def load_model():
    model_repo = "iamthearafatkhan/distilbert-spam2336"
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()


# PREDICTION FUNCTION

def predict(message):
    inputs = tokenizer(
        message,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred = np.argmax(probs)

    return pred, probs


# UI HEADER

st.markdown("<h1 style='text-align:center;'>📨 Spam Email Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:17px;'>Powered by DistilBERT Model</p>", unsafe_allow_html=True)



# ---- Transparent textarea  ----
st.markdown("""
<style>
/* All Streamlit textarea containers */
[data-testid="stTextArea"] > div > div > textarea {
    background-color: rgba(255, 255, 255, 0.35) !important;
    color: black !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    backdrop-filter: blur(4px);
}

/* Remove white background from parent containers */
[data-testid="stTextArea"] > div,
[data-testid="stTextArea"] > div > div {
    background: transparent !important;
}

/*label color */
label[data-testid="stWidgetLabel"] {
    color: white !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)


# TEXT INPUT

user_input = st.text_area(
    "✉️ Type your message:",
    placeholder="Write or paste email text here...",
    height=120
)




# FILE UPLOAD INPUT

st.subheader("📑Upload a Text File")
uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])

file_content = None
if uploaded_file is not None:
    file_content = uploaded_file.read().decode("utf-8")
    st.text_area("File Content:", file_content, height=150)


# RUN PREDICTION

if st.button("🔍 Predict"):
    text = user_input if uploaded_file is None else file_content

    if text and text.strip():
        with st.spinner("Analyzing message..."):
            label, probs = predict(text)
            result = "📬 HAM (Not Spam)" if label == 0 else "🚨 SPAM"

        st.markdown("---")
        st.subheader("Prediction Result:")
        st.success(result)
        st.metric("Confidence", f"{probs[label]*100:.2f}%")

        st.markdown("### Class Probabilities")
        st.progress(float(probs[1]))
        st.write(f"**HAM:** {probs[0]*100:.2f}%")
        st.write(f"**SPAM:** {probs[1]*100:.2f}%")
    else:
        st.warning("⚠️ Please type a message or upload a file.")

st.markdown("---")
st.markdown("<p style='text-align:center;'>Built with ❤️ using Streamlit + DistilBERT</p>", unsafe_allow_html=True)
