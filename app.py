import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

st.set_page_config(page_title="AI News Summarizer", page_icon="📰")
st.title(" AI News Summarizer")


# Direct Model Loading (No pipeline task string needed)
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


try:
    model, tokenizer = load_model_and_tokenizer()
    st.success("Model ready to go!")
except Exception as e:
    st.error(f"Initialization Error: {e}")

text_input = st.text_area("Paste news article here:", height=250)

if st.button("Summarize Now"):
    if text_input:
        with st.spinner('Summarizing...'):
            # Manual Tokenization & Generation
            inputs = tokenizer([text_input], max_length=1024, return_tensors="pt", truncation=True)
            summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, min_length=40)
            summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
                0]

            st.subheader("Summary Result:")
            st.info(summary)
    else:
        st.warning("Please enter some text.")
