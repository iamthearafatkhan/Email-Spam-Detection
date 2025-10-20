import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import base64
from pathlib import Path

# --- App Config ---
st.set_page_config(
    page_title="Spam Detector | DistilBERT",
    page_icon="📧",
    layout="centered"
)

# --- Background Setup ---
def set_bg(image_file):
    """Set background image using base64 encoding"""
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
    background-attachment: fixed;
}}

div.block-container {{
    backdrop-filter: blur(8px);
    background-color: rgba(255,255,255,0.78);
    border-radius: 18px;
    padding: 30px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    color: #000000; /* <-- Make all text inside container black */
}}

h1, h2, h3, h4, h5, h6, p, label, span, div {{
    color: #000000 !important; /* <-- Force all headings and labels to black */
}}

.stButton>button {{
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 8px 18px;
    font-weight: 600;
}}

.stButton>button:hover {{
    background-color: #1d4ed8;
}}

</style>
            """,
            unsafe_allow_html=True
        )

set_bg("bgb.jpg")

# --- Load Model and Tokenizer (cached for performance) ---
"""
@st.cache_resource
def load_model():
    model_path = "./distilbert_spam_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device 
    """

@st.cache_resource
def load_model():
    model_repo = "iamthearafatkhan/distilbert-spam2336"  # ✅ use your repo name
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


tokenizer, model, device = load_model()

# --- UI ---
st.markdown("<h1 style='text-align:center;'>📨 Spam E-Mail Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:17px;'>Powered by <b>DistilBERT</b> NLP model</p>", unsafe_allow_html=True)

user_input = st.text_area(
    "✉️ Type your message below:",
    placeholder="e.g., Congratulations! You've won a free iPhone! 🎉",
    height=120
)

if st.button("🔍 Predict"):
    if user_input.strip():
        with st.spinner("Analyzing message... Please wait"):
            inputs = tokenizer(
                user_input,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
                pred = np.argmax(probs)
                label = "📬 HAM" if pred == 0 else "🚨 SPAM"

            st.markdown("---")
            st.subheader("Prediction Result:")
            st.success(f"**{label}**")
            st.metric(label="Confidence", value=f"{probs[pred]*100:.2f}%")

            st.markdown("### Class Probabilities")
            st.progress(float(probs[1]))
            st.write(f"**HAM:** {probs[0]*100:.2f}%")
            st.write(f"**SPAM:** {probs[1]*100:.2f}%")
    else:
        st.warning("⚠️ Please enter a message first!")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ using <b>Streamlit</b> + <b>DistilBERT</b></p>",
    unsafe_allow_html=True
)
