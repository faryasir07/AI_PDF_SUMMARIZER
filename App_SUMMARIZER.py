import streamlit as st

# Set page config FIRST before any other Streamlit commands
st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="centered"
)

# Now the rest of the imports
import json
import requests
from PyPDF2 import PdfReader
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load template
with open('template.json') as f:
    TEMPLATE = json.load(f)

# Load model and tokenizer only once
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("D:\TinyLlama_model")
    model = AutoModelForCausalLM.from_pretrained(
        "D:\TinyLlama_model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def extract_text_from_pdf(url):
    response = requests.get(url)
    pdf_file = BytesIO(response.content)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text[:3000]  # Truncate to fit model context

def generate_summary(model, tokenizer, text, audience, length):
    # Format prompt using template
    prompt = TEMPLATE["prompt_template"].format(
        audience=audience,
        length=f"{length} sentences",
        text=text
    )
    
    # Format messages for chat model
    messages = [
        {"role": "system", "content": TEMPLATE["system_prompt"]},
        {"role": "user", "content": prompt}
    ]
    
    # Tokenize and generate
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    # Decode and remove special tokens
    summary = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # Remove <|assistant|> prefix if present
    if summary.startswith("<|assistant|>"):
        summary = summary.replace("<|assistant|>", "", 1).strip()
    
    return summary

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stSlider>div>div>div>div {
        background-color: rgba(255,255,255,0.9) !important;
        color: #000000 !important;  /* Changed text color to black */
    }
    .css-1d391kg {padding-top: 3rem;}
    h1, h2, h3, .stMarkdown {color: white !important;}
    .stButton>button {
        background: #ff4b4b !important;
        color: white !important;
        border: none;
        padding: 10px 24px;
        border-radius: 50px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: #ff2b2b !important;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📚 Research Paper Summarizer")
st.markdown("### AI-powered summaries for academic papers using TinyLlama")

# Inputs
col1, col2 = st.columns(2)
with col1:
    url = st.text_input("PDF URL:", placeholder="https://arxiv.org/pdf/...")
    
with col2:
    audience = st.selectbox(
        "Summary Type:",
        ("Beginner-Friendly", "Technical", "Concise", "Detailed"),
        index=0
    )

length = st.slider("Summary Length (sentences):", 3, 10, 5)

# Generate button
if st.button("Generate Summary", use_container_width=True):
    if not url:
        st.warning("Please enter a PDF URL")
    else:
        with st.spinner("Downloading and processing paper..."):
            try:
                text = extract_text_from_pdf(url)
                if not text:
                    st.error("Failed to extract text from PDF")
                    st.stop()
                
                tokenizer, model = load_model()
                
                with st.spinner("Generating summary (this may take 20-30 seconds)..."):
                    summary = generate_summary(model, tokenizer, text, audience, length)
                
                st.success("Summary Generated!")
                st.subheader("Research Summary")
                st.markdown(f"<div style='background:rgba(255,255,255,0.1); padding:20px; border-radius:10px;'>{summary}</div>", 
                           unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with ♥ using TinyLlama 1.1B")