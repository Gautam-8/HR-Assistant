import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Constants
VECTOR_DIR = "vector_store"
UPLOAD_DIR = "data/hr_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Embeddings from LM Studio
class LMStudioEmbeddings(Embeddings):
    def __init__(self, endpoint_url: str = "http://localhost:1234/v1/embeddings", model_name: str = "text-embedding-nomic-embed-text-v1.5"):
        self.endpoint_url = endpoint_url
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model_name, "input": texts}
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.endpoint_url, json=payload, headers=headers)
        response.raise_for_status()
        return [item['embedding'] for item in response.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Chunking
def split_documents(documents):
    documents = [doc for doc in documents if doc.page_content.strip()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

# Load HR documents
def load_hr_docs(file_path):
    if file_path.endswith(".pdf"):
        return PyMuPDFLoader(file_path).load()
    elif file_path.endswith(".docx") or file_path.endswith(".doc"):
        return Docx2txtLoader(file_path).load()
    elif file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    else:
        st.error(f"Unsupported file type: {file_path}")
        return []

# Embed and store in Chroma
def embed_and_store(chunks):
    embeddings = LMStudioEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DIR
    )
    vectordb.persist()
    return vectordb

# Load existing vector DB
def load_existing_vectorstore():
    try:
        embeddings = LMStudioEmbeddings()
        return Chroma(
            embedding_function=embeddings,
            persist_directory=VECTOR_DIR
        )
    except Exception:
        return None

# ────────────────────────────────────────────────────────────────────────────
# Streamlit UI Starts Here
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="HR Assistant", page_icon=":robot_face:")
st.title("🤖 HR Assistant")

vectordb = None

# Upload section
uploaded_file = st.file_uploader("📤 Upload HR File", type=["pdf", "docx", "txt"])

if uploaded_file:
    saved_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"📁 Saved to {saved_path}")

    docs = load_hr_docs(saved_path)
    st.success(f"📄 Loaded {len(docs)} documents")

    with st.spinner("🔄 Splitting and embedding..."):
        chunks = split_documents(docs)
        if chunks:
            st.success(f"🧩 Split into {len(chunks)} chunks")
            vectordb = embed_and_store(chunks)
            st.success("✅ Embedded and stored!")
        else:
            st.error("❌ No valid chunks found in this document.")
            st.stop()
else:
    vectordb = load_existing_vectorstore()
    if vectordb:
        st.info("✅ Loaded existing embedded documents.")
    else:
        st.warning("⚠️ No documents found. Upload to get started.")

# Always show question input
st.markdown("---")
st.subheader("💬 Ask an HR Question")

user_q = st.text_input("Type your HR-related question here:")

if user_q and vectordb:
    with st.spinner("🔍 Searching and thinking..."):
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            model="google/gemma-3-4b",
            api_key=SecretStr("not-needed")
        )
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        result = qa(user_q)

        st.subheader("📝 Answer")
        st.write(result['result'])

        st.subheader("📚 Sources Used")
        for i, doc in enumerate(result['source_documents']):
            st.markdown(f"**Source {i+1}:**")
            st.text(doc.page_content[:300] + "...")
elif user_q:
    st.warning("❌ No documents embedded yet. Please upload a file to begin.")
