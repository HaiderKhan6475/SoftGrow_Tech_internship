&#x20;AI Multi-Project Suite (Streamlit Dashboard)



A comprehensive AI-powered dashboard that integrates three distinct machine learning and NLP projects into a single, user-friendly interface. Created as part of the \*\*SoftGrow Tech Internship\*\*.



\---



\## 🌟 Featured AI Projects



\### 1. 📰 AI News Summarizer

This module uses the \*\*BART (Large-CNN)\*\* model from HuggingFace to summarize long articles.

\- \*\*Why it's cool:\*\* It can take a 1000-word article and turn it into a 50-word summary in seconds.

\- \*\*Tech:\*\* Transformers, PyTorch, AutoTokenizer.



\### 2. 📄 CV-Based Personality Predictor

An automated tool that parses PDF resumes to analyze candidate personality traits.

\- \*\*Traits Analyzed:\*\* Openness, Extraversion, and Conscientiousness.

\- \*\*How it works:\*\* It uses keyword frequency analysis and text parsing to score traits based on professional experience.

\- \*\*Tech:\*\* PyPDF2, NLP Keyword Matching.



\### 3. 🔬 Malaria Parasite Detector

A medical imaging tool that identifies malaria-infected cells from microscopic slides.

\- \*\*Model:\*\* Convolutional Neural Network (CNN) saved as `malaria\_model.h5`.

\- \*\*Accuracy:\*\* High-precision detection between Parasitized and Uninfected cells.

\- \*\*Tech:\*\* TensorFlow, Keras, Pillow (PIL).



\---



\## ⚙️ Installation \& Usage Guide



Follow these steps to run the project on your local machine:



\### 1. Clone the Project

```bash

git clone https://github.com

cd Ai\_SGT\_projects

Use code with caution.



2\. Create Virtual Environment

bash

python -m venv .venv

.venv\\Scripts\\activate  # For Windows

Use code with caution.



3\. Install All Requirements

bash

pip install streamlit tensorflow pillow numpy PyPDF2 transformers torch

Use code with caution.



4\. Run the Dashboard

bash

streamlit run main\_dashboard.py

Use code with caution.   



📂 Repository Structure

main\_dashboard.py: The main Streamlit entry point.

malaria\_model.h5: Pre-trained model for the Malaria Detector.

.gitignore: Configured to exclude heavy venv and cell\_images folders.

README.md: Project documentation.

👨‍💻 Developer Information

Name: Haider Khan       

Email: thinkcode013@gmail.com

Role: AI \& DevOps Intern

Company: SoftGrow Tech for internship

