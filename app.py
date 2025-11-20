import streamlit as st
import pdfplumber
from PIL import Image
import pytesseract
from io import BytesIO
from transformers import pipeline
import tempfile

st.set_page_config(page_title="AI Study Notes Generator", layout="wide")
st.title("📚 AI Study Notes Generator")

# Initialize Hugging Face pipelines (open-source)
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="google/flan-t5-base")
    qg_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    return summarizer, qg_pipeline

summarizer, qg_pipeline = load_models()

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                # If page has no text, try OCR
                img = page.to_image(resolution=300).original
                page_text = pytesseract.image_to_string(img)
                text += page_text + "\n"
    return text

def generate_notes(text, max_length=512):
    summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']
    # Convert summary into bullet points
    bullets = "\n".join([f"- {line.strip()}" for line in summary.split(". ") if line.strip()])
    return bullets

def generate_mcqs(text, num_questions=5):
    mcqs = ""
    for i in range(num_questions):
        prompt = f"Generate a multiple choice question from the following text:\n{text}\nQ{i+1}:"
        result = qg_pipeline(prompt, max_length=150)[0]['generated_text']
        mcqs += f"{i+1}. {result}\n\n"
    return mcqs

def generate_flashcards(text, num_cards=5):
    flashcards = ""
    for i in range(num_cards):
        prompt = f"Create a flashcard (Question and Answer) from the following text:\n{text}\nFlashcard {i+1}:"
        result = qg_pipeline(prompt, max_length=100)[0]['generated_text']
        flashcards += f"{i+1}. {result}\n\n"
    return flashcards

# Streamlit UI
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ Text extracted!")

    st.subheader("Generated Notes")
    notes = generate_notes(raw_text)
    st.text_area("Notes (Bullet Points)", notes, height=300)

    st.subheader("Generated MCQs")
    mcqs = generate_mcqs(raw_text)
    st.text_area("MCQs", mcqs, height=300)

    st.subheader("Generated Flashcards")
    flashcards = generate_flashcards(raw_text)
    st.text_area("Flashcards", flashcards, height=300)

    # Save output as TXT
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        output_content = f"--- Notes ---\n{notes}\n\n--- MCQs ---\n{mcqs}\n\n--- Flashcards ---\n{flashcards}"
        tmp_file.write(output_content.encode('utf-8'))
        st.download_button("📥 Download All as TXT", tmp_file.name, file_name="study_notes.txt")
