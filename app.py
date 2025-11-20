import streamlit as st
import pdfplumber
import pytesseract
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from gpt4all import GPT4All
import os

st.set_page_config(page_title="AI Study Notes Generator", layout="wide")
st.title("📝 AI Study Notes Generator")

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    st.info("Processing your PDF...")
    text_content = ""
    
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text_content += page.extract_text() or ""
    
    st.success("PDF loaded successfully!")

    # Show a snippet
    st.text_area("Extracted Text (preview)", text_content[:1000], height=200)

    # ----------------------------
    # Initialize AI model
    # ----------------------------
    model_path = "ggml-gpt4all-j-v1.3-groovy.bin"  # Download model from GPT4All official repo
    if not os.path.exists(model_path):
        st.warning(f"Please download the model from GPT4All and place it at: {model_path}")
    else:
        st.info("Initializing AI model...")
        llm = GPT4All(model_path)
        
        # ----------------------------
        # Ask for notes summary
        # ----------------------------
        prompt = st.text_area("Enter your prompt for notes", "Summarize the uploaded text in bullet points")
        if st.button("Generate Notes"):
            with st.spinner("Generating notes..."):
                response = llm.generate(prompt + "\n\n" + text_content[:5000])  # Limit to first 5000 chars
                st.success("✅ Notes generated!")
                st.text_area("AI Notes", response, height=400)
