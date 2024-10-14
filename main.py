import os
import langchain
import pickle
import time
import streamlit as st
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader

# Streamlit Title
st.title("📰 News Research Tool")

# Sidebar for API Key Configuration
st.sidebar.title("API KEY CONFIGURATION")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Fixed values for temperature and max tokens
temperature = 0.8
max_tokens = 1000

# Initialize LLM with OpenAI if the API key is provided
llm = None
if api_key:
    try:
        llm = OpenAI(api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        st.error(f"⚠️ Error initializing OpenAI client: {e}")

# Sidebar for URL Input
st.sidebar.title("🔗 Enter News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_button_clicked = st.sidebar.button("Process URLs")
vector_store_path = "vector_index"

main_placeholder = st.empty()

if process_button_clicked:
    if llm:  # Ensure llm is initialized before proceeding
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("🔄 Data Loading...")
        data = loader.load()
        
        # Split data
        splitter = RecursiveCharacterTextSplitter(separators=["\n", "\n\n", ",", " "], chunk_size=1000, chunk_overlap=200)
        main_placeholder.text("✂️ Splitting the Data into chunks...")
        docs = splitter.split_documents(data)
        
        # Create embeddings only if API key is valid
        if api_key:
            try:
                embeddings = OpenAIEmbeddings(api_key=api_key)  # Pass the API key here
                vector_index = FAISS.from_documents(docs, embeddings)
                main_placeholder.text("🔧 Creating embeddings and building vector index...")
                time.sleep(2)
                vector_index.save_local(vector_store_path)

                st.success("✅ Data processed successfully.")
                time.sleep(2)
            except Exception as e:
                st.error(f"⚠️ Error creating embeddings: {str(e)}")
        else:
            st.error("⚠️ Please enter a valid OpenAI API key.")
    else:
        st.error("⚠️ Please enter a valid OpenAI API key.")

# Load FAISS index if available   
if os.path.exists(vector_store_path):
    try:
        embeddings = OpenAIEmbeddings(api_key=api_key)  # Ensure to pass the API key
        vector_index = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        pass

# Question input for querying the FAISS index
query = st.text_input("💬 Enter your question:")

# Process the query if provided
if query:
    if vector_index is not None:
        try:
            # Set up the Retrieval QA chain
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_index.as_retriever())
            
            # Execute the query
            result = chain({"question": query}, return_only_outputs=True)
            
            # Display the answer
            st.header("🔍 Answer")
            st.markdown(f"<p style='font-size:22px;'>{result['answer']}</p>", unsafe_allow_html=True)
            
            # Display the sources
            sources = result.get("sources", "")
            if sources:
                st.subheader("📚 Sources")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
            
        except Exception as e:
            st.error(f"🚨 An error occurred during query processing: {str(e)}")
    else:
        st.error("⚠️ FAISS index is not available. Please process the URLs first.")
