# -*- coding: utf-8 -*-





# HR Chatbot Web App using Streamlit + LangChain + Static Files from GitHub

import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(repo_id="...", huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")) 


# ---------------- Setup ---------------- #
st.set_page_config(page_title="CISF Chatbot", layout="wide")
st.title("🤖 CISF Chatbot - Ask Your Query")

# ---------------- Load Files from data/ folder ---------------- #
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

documents = []
for filename in os.listdir(data_dir):
    path = os.path.join(data_dir, filename)
    try:
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(path)
        else:
            continue
        file_docs = loader.load()
        documents.extend(file_docs)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
st.write("Loaded documents:", len(documents))
# ---------------- Build Vector Store ---------------- #
if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        st.error("❌ No text chunks created. Adjust chunk size or check document content.")
        st.stop()

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"❌ Error loading embedding model: {e}")
        st.stop()

    try:
        db = FAISS.from_documents(chunks, embeddings)
    except IndexError:
        st.error("❌ No embeddings could be generated. Check if documents are valid and non-empty.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error creating FAISS vector store: {e}")
        st.stop()

    retriever = db.as_retriever()

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        st.error("❌ Hugging Face API token not found. Add it to Streamlit secrets or a .env file.")
        st.stop()

    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        temperature=0.3,
        model_kwargs={"max_length": 512},
        huggingfacehub_api_token=token
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.subheader("💬 Ask a question")
    query = st.text_input("Type your question here...")

    if query:
        with st.spinner("🤔 Thinking..."):
            response = qa_chain.invoke({"query": query})
        st.success("Answer")
        st.write(response)
else:
    st.warning("No documents found. Please add files to the 'data/' folder in your GitHub repo.")

