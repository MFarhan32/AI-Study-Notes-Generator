import streamlit as st
from pypdf import PdfReader
from transformers import pipeline

# --------------------------
# 🔥 Page Config
# --------------------------
st.set_page_config(
    page_title="AI Study Notes Generator",
    page_icon="📘",
    layout="wide",
)

# --------------------------
# 🎨 Custom CSS for UI
# --------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    text-align: center;
    font-weight: bold;
    background: -webkit-linear-gradient(90deg, #4e8cff, #8e54e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}
.box {
    padding: 20px;
    border-radius: 10px;
    background-color: #f5f6fa;
    border: 1px solid #e3e5e8;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# 💬 Heading
# --------------------------
st.markdown("<h1 class='main-title'>AI Study Notes Generator</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Upload your PDF → Get Notes, MCQs, Flashcards instantly</p>", unsafe_allow_html=True)


# --------------------------
# ⚙ Load Model (HuggingFace Pipeline)
# --------------------------
@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base")

llm = load_llm()


# --------------------------
# 📄 PDF Text Extractor
# --------------------------
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# --------------------------
# 🤖 AI Prompt Functions
# --------------------------
def generate_notes(text):
    prompt = f"Summarize this text into clear study notes:\n\n{text}"
    return llm(prompt, max_length=600)[0]["generated_text"]

def generate_mcqs(text):
    prompt = f"Generate 5 MCQs with answers from this text:\n\n{text}"
    return llm(prompt, max_length=700)[0]["generated_text"]

def generate_flashcards(text):
    prompt = f"Generate 10 flashcards (Q/A format) from this text:\n\n{text}"
    return llm(prompt, max_length=700)[0]["generated_text"]


# --------------------------
# 📤 File Upload
# --------------------------
st.subheader("📄 Upload PDF File")
uploaded_file = st.file_uploader("Choose your study material PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_pdf_text(uploaded_file)

    st.success("PDF extracted successfully!")

    st.markdown("---")
    st.subheader("⚙ Select What You Want to Generate")

    col1, col2, col3 = st.columns(3)
    generate_notes_opt = col1.checkbox("📝 Notes")
    generate_mcqs_opt = col2.checkbox("❓ MCQs")
    generate_flashcards_opt = col3.checkbox("⚡ Flashcards")

    st.markdown("---")

    if st.button("🚀 Generate AI Content", use_container_width=True):
        with st.spinner("AI is generating your content... This may take a minute ⏳"):

            notes_output = mcq_output = flash_output = None

            if generate_notes_opt:
                notes_output = generate_notes(pdf_text)

            if generate_mcqs_opt:
                mcq_output = generate_mcqs(pdf_text)

            if generate_flashcards_opt:
                flash_output = generate_flashcards(pdf_text)

        st.success("🎉 Content generated successfully!")

        # Results Display
        if notes_output:
            st.markdown("### 📝 Study Notes")
            st.markdown(f"<div class='box'>{notes_output}</div>", unsafe_allow_html=True)

        if mcq_output:
            st.markdown("### ❓ MCQs")
            st.markdown(f"<div class='box'>{mcq_output}</div>", unsafe_allow_html=True)

        if flash_output:
            st.markdown("### ⚡ Flashcards")
            st.markdown(f"<div class='box'>{flash_output}</div>", unsafe_allow_html=True)

        # Downloads
        st.markdown("---")
        st.subheader("📥 Download Your Results")

        if notes_output:
            st.download_button("Download Notes", notes_output, "notes.txt")

        if mcq_output:
            st.download_button("Download MCQs", mcq_output, "mcqs.txt")

        if flash_output:
            st.download_button("Download Flashcards", flash_output, "flashcards.txt")
