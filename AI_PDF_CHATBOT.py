import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os
import requests
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Local PDF Chatbot", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🤖 Local PDF Chatbot")
st.caption("Chat with PDF documents using local AI models")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Embedding Settings")
    embedding_model = st.text_input(
        "Embedding Model", 
        value="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model name"
    )
    
    st.subheader("LLM Settings")
    llm_model = st.text_input(
        "LLM Model Name", 
        value="ai/gemma3:1B-Q4_K_M",
        help="Model name for local inference"
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
    max_tokens = st.slider("Max Tokens", 128, 4096, 1024)
    
    st.subheader("API Configuration")
    base_url = st.text_input(
        "Base URL", 
        value="http://localhost:12434/engines/v1",
        help="Local inference server URL"
    )
    api_key = st.text_input(
        "API Key", 
        value="docker",
        type="password",
        help="API key for local inference server"
    )
    
    pdf_url = st.text_input("PDF URL", placeholder="https://example.com/document.pdf")
    process_btn = st.button("Process PDF")
    
    if not pdf_url:
        st.warning("Please enter a PDF URL")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# PDF processing function with progress indicators
def process_pdf(url, embedding_model):
    try:
        with st.spinner("Downloading PDF..."):
            response = requests.get(url)
            response.raise_for_status()  # Raise error for bad status
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        with st.spinner("Loading and splitting PDF..."):
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(pages)
        
        with st.spinner("Creating embeddings (this may take a while)..."):
            # Create embeddings with local model
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            
            # Create vector store
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=None
            )
            
        os.unlink(tmp_file_path)  # Delete temporary file
        return vector_store
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

# RAG processing function with timeout handling
def get_rag_response(query, vector_store, llm_model, base_url, api_key, temperature, max_tokens):
    try:
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Define prompt template
        template = """Answer the question based ONLY on the following context. 
        Do NOT guess or use external knowledge. If you don't know, say so.
        
        Context: {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Creating RAG chain with local LLM
        llm = ChatOpenAI(
            model=llm_model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            max_retries=3,  # Adding retries for stability
            request_timeout=120  # Increasing timeout
        )
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain.stream(query)
    
    except Exception as e:
        st.error(f"Error in RAG pipeline: {str(e)}")
        return [f"Error: {str(e)}"]

# Process PDF when button is clicked
if process_btn and pdf_url:
    with st.spinner(f"Processing PDF with {embedding_model}..."):
        start_time = time.time()
        st.session_state.vector_store = process_pdf(pdf_url, embedding_model)
        if st.session_state.vector_store:
            st.session_state.pdf_processed = True
            process_time = time.time() - start_time
            st.success(f"PDF processed successfully in {process_time:.1f} seconds!")

# Chat input and processing
if prompt := st.chat_input("Ask a question about the PDF (type /bye to exit)"):
    # Check for exit commands
    if prompt.lower() in ["/bye", "/exit"]:
        st.session_state.messages.append({"role": "assistant", "content": "Goodbye! Feel free to start a new chat anytime."})
        st.rerun()
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process question if vector store exists
    if not st.session_state.get("vector_store"):
        response = "Please process a PDF first using the sidebar configuration"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        try:
            # Display assistant response with streaming
            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response = ""
                
                # Stream response chunks with timeout handling
                stream = get_rag_response(
                    prompt, 
                    st.session_state.vector_store,
                    llm_model,
                    base_url,
                    api_key,
                    temperature,
                    max_tokens
                )
                
                # Stream with timeout handling
                start_time = time.time()
                for chunk in stream:
                    # Check for timeout
                    if time.time() - start_time > 120:  # 2-minute timeout
                        st.warning("Response timed out")
                        break
                        
                    full_response += chunk
                    response_container.markdown(full_response + "▌")
                    time.sleep(0.02)  # Smooth streaming
                
                response_container.markdown(full_response)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Add reset button in sidebar
with st.sidebar:
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("Clear PDF Cache"):
        st.session_state.vector_store = None
        st.session_state.pdf_processed = False
        st.success("PDF cache cleared!")