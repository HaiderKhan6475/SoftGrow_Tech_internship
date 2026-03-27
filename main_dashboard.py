import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import PyPDF2
# Direct Model Classes (No pipeline dependency)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="AI Multi-Project Suite", layout="wide")

# Sidebar
st.sidebar.title(" Project Selection")
choice = st.sidebar.radio("Go to:", ["News Summarizer", "Personality Predictor", "Malaria Detector"])

# --- PROJECT 1: SUMMARIZER ---
if choice == "News Summarizer":
    st.header(" AI News Summarizer")


    @st.cache_resource
    def load_summarizer():
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return model, tokenizer


    model, tokenizer = load_summarizer()
    text = st.text_area("Paste News Article:", height=200)

    if st.button("Summarize"):
        if text:
            with st.spinner("Processing..."):
                inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
                summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=130, min_length=30)
                summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
                st.success(summary[0])
        else:
            st.warning("Please enter text!")

# --- PROJECT 2: PERSONALITY ---
elif choice == "Personality Predictor":
    st.header(" Personality Prediction from CV")
    uploaded_cv = st.file_uploader("Upload CV (PDF)", type="pdf")
    if uploaded_cv:
        reader = PyPDF2.PdfReader(uploaded_cv)
        cv_text = "".join([page.extract_text() for page in reader.pages]).lower()
        traits = {
            "Openness": ["creative", "innovative", "research", "design"],
            "Extraversion": ["leadership", "team", "social", "public speaking"],
            "Conscientiousness": ["organized", "managed", "strategic", "planning"]
        }
        st.subheader("Results:")
        for trait, keys in traits.items():
            score = sum(1 for k in keys if k in cv_text)
            st.write(f"**{trait}:** {'⭐' * score} ({score} matches)")

# --- PROJECT 3: MALARIA ---
elif choice == "Malaria Detector":
    st.header("🔬 Malaria Parasite Detection")
    # Load your freshly trained model
    try:
        malaria_model = tf.keras.models.load_model('malaria_model.h5')
        st.success("Trained Model Loaded!")
    except:
        st.error("Model file 'malaria_model.h5' not found. Please train it first.")

    img_file = st.file_uploader("Upload Cell Image", type=["png", "jpg", "jpeg"])
    if img_file:
        img = Image.open(img_file).resize((64, 64))
        st.image(img, caption="Microscopic View")

        if st.button("Analyze Image"):
            img_arr = np.array(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            prediction = malaria_model.predict(img_arr)
            # Threshold check
            if prediction[0][0] < 0.5:
                st.error(" Result: Parasitized (Malaria Detected)")
            else:
                st.success(" Result: Uninfected (Healthy Cell)")
