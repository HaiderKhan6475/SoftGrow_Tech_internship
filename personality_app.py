import streamlit as st
import PyPDF2
import spacy
from collections import Counter

# Load the NLP model we just installed
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="CV Personality AI", page_icon="🧠")
st.title("AI  Personality Prediction (CV Analysis)")

# Define Personality Traits with weighted keywords
trait_dictionary = {
    "Openness": ["creative", "innovative", "research", "design", "learning", "curious", "flexible"],
    "Conscientiousness": ["organized", "managed", "strategic", "efficiency", "discipline", "detail", "planned"],
    "Extraversion": ["leadership", "team", "communication", "social", "outspoken", "network", "public"],
    "Agreeableness": ["supportive", "teamwork", "empathy", "patient", "cooperative", "helpful", "trust"],
    "Emotional Stability": ["calm", "resilient", "stable", "composed", "stress", "focused"]
}


def analyze_personality(text):
    doc = nlp(text.lower())
    scores = {trait: 0 for trait in trait_dictionary}

    # Analyze tokens and match with our dictionary
    for token in doc:
        for trait, keywords in trait_dictionary.items():
            if token.text in keywords or token.lemma_ in keywords:
                scores[trait] += 1
    return scores


# UI Section
uploaded_file = st.file_uploader("Upload a Resume/CV (PDF format)", type="pdf")

if uploaded_file:
    with st.spinner("Analyzing CV content..."):
        # Extract text from PDF
        reader = PyPDF2.PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()

        # Get results
        trait_scores = analyze_personality(full_text)

        st.success("Analysis Complete!")

        # Display Results in a nice format
        cols = st.columns(2)
        for i, (trait, score) in enumerate(trait_scores.items()):
            with cols[i % 2]:
                st.metric(label=trait, value=f"{score} matches")
                st.progress(min(score * 10, 100))  # Progress bar (max 10 matches = 100%)

        st.divider()
        st.write("### Top Career Suggestions based on Traits:")
        top_trait = max(trait_scores, key=trait_scores.get)
        if trait_scores[top_trait] > 0:
            st.write(
                f"Based on your high **{top_trait}** score, you might excel in roles that require strong professional alignment in this area.")
        else:
            st.warning("Not enough keywords found to give a solid career advice. Try a more detailed CV!")

