import os
import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader

# === PAGE CONFIGURATION ===
st.set_page_config(page_title="URL Insight Assistant", layout="centered", page_icon="🔍")

# === APP TITLE & SUBTITLE ===
st.title("🔍 URL Insight Assistant")
st.markdown("<h5 style='color:#a0a0a0; font-weight:normal;'>Process any webpage URL to extract, index, and query its content efficiently.</h5>", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.title("🔐 API Key")
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""

api_key_input = st.sidebar.text_input(
    "Enter your Google Generative AI API key", 
    type="password", 
    value=st.session_state.google_api_key
)

# Save API key in session state to persist until browser/tab closed
if api_key_input != st.session_state.google_api_key:
    st.session_state.google_api_key = api_key_input

st.sidebar.title("🌐 Webpage URLs (1 or more)")
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]

process_button = st.sidebar.button("⚙️ Process URLs")

vector_store_path = "vector_index"
main_placeholder = st.empty()

# === INITIALIZE LLM & EMBEDDINGS ===
llm, embeddings = None, None
if st.session_state.google_api_key:
    os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
    try:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash-001",
            temperature=0.8,
            max_output_tokens=1000,
            # Custom prompt template to avoid prefixes like "FINAL ANSWER:"
            prompt_template=(
                "You are a helpful assistant. "
                "Answer the question based only on the provided context. "
                "Be concise and avoid extra commentary or prefixes."
            )
        )
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini: {e}")
else:
    st.info("🔑 Please enter your Google Generative AI API key to continue.")

# === PROCESS URLS INTO VECTOR INDEX ===
if process_button:
    if not st.session_state.google_api_key:
        st.error("❌ Please provide your Google Generative AI API key before processing URLs.")
    else:
        valid_urls = [url.strip() for url in urls if url.strip()]
        if not valid_urls:
            st.warning("⚠️ Please enter at least one valid URL to process.")
        else:
            try:
                main_placeholder.info("🔄 Loading content from the provided URL(s)...")
                loader = UnstructuredURLLoader(urls=valid_urls)
                documents = loader.load()

                main_placeholder.info("✂️ Splitting content into manageable chunks...")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                chunks = splitter.split_documents(documents)

                main_placeholder.info("🔧 Generating embeddings and building the index...")
                vector_index = FAISS.from_documents(chunks, embeddings)
                vector_index.save_local(vector_store_path)

                st.success("✅ Content indexed successfully! You can now ask questions below.")
            except Exception as e:
                st.error(f"🚨 Error while processing URLs: {e}")

# === LOAD EXISTING VECTOR INDEX IF AVAILABLE ===
vector_index = None
if os.path.exists(vector_store_path) and st.session_state.google_api_key:
    try:
        vector_index = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"❌ Failed to load vector index: {e}")

# === QUERY INPUT ===
query = st.text_input("💬 Ask a question about the processed content:")

# === QUERY HANDLING ===
if query:
    if not st.session_state.google_api_key:
        st.warning("⚠️ Please enter your API key to ask questions.")
    elif vector_index is None:
        st.warning("⚠️ Please process at least one URL first before querying.")
    else:
        try:
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vector_index.as_retriever()
            )
            result = chain({"question": query}, return_only_outputs=True)

            answer = result.get("answer", "").replace("FINAL ANSWER:", "").strip()

            st.header("🧠 Answer")
            st.markdown(f"<p style='font-size:22px;'>{answer}</p>", unsafe_allow_html=True)

            sources = result.get("sources", "").strip()
            if sources:
                st.subheader("📚 Sources")
                for src in sources.split("\n"):
                    if src.strip():
                        st.markdown(f"- {src.strip()}")

        except Exception as e:
            st.error(f"🚨 Error while answering query: {e}")
