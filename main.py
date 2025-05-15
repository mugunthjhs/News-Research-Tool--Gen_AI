import os
import time
import streamlit as st
from langchain_openai import OpenAI, OpenAIEmbeddings
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
        st.error(f"⚠️ Error initializing OpenAI client: {str(e)}")

# Sidebar for URL Input
st.sidebar.title("🔗 Enter News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_button_clicked = st.sidebar.button("Process URLs")
vector_store_path = "vector_index"

main_placeholder = st.empty()

if process_button_clicked:
    if llm:
        try:
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("🔄 Data Loading...")
            data = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                separators=["\n", "\n\n", ",", " "],
                chunk_size=1000,
                chunk_overlap=200
            )
            main_placeholder.text("✂️ Splitting the Data into chunks...")
            docs = splitter.split_documents(data)

            embeddings = OpenAIEmbeddings(api_key=api_key)
            vector_index = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("🔧 Creating embeddings and building vector index...")
            time.sleep(2)
            vector_index.save_local(vector_store_path)

            st.success("✅ Data processed successfully.")
        except Exception as e:
            st.error(f"⚠️ Error during processing: {str(e)}")
    else:
        st.error("⚠️ Please enter a valid OpenAI API key.")

# Load FAISS index if available
vector_index = None
if os.path.exists(vector_store_path):
    try:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vector_index = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"⚠️ Error loading vector index: {str(e)}")

# Question input for querying the FAISS index
query = st.text_input("💬 Enter your question:")

# Process the query if provided
if query:
    if vector_index is not None and llm is not None:
        try:
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vector_index.as_retriever()
            )

            result = chain({"question": query}, return_only_outputs=True)

            st.header("🔍 Answer")
            st.markdown(f"<p style='font-size:22px;'>{result['answer']}</p>", unsafe_allow_html=True)

            sources = result.get("sources", "")
            if sources:
                st.subheader("📚 Sources")
                for source in sources.split("\n"):
                    st.write(source)

        except Exception as e:
            st.error(f"🚨 An error occurred during query processing: {str(e)}")
    else:
        st.error("⚠️ FAISS index is not available or LLM not initialized. Please process the URLs first.")
