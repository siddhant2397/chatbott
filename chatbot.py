import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Set up page
st.set_page_config(page_title="📘 CISF Chatbot", layout="wide")
st.title("🤖 CISF Chatbot - Ask Your Query")

# Load files
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
        docs = loader.load()
        documents.extend(docs)
    except Exception as e:
        st.error(f"❌ Failed to load {filename}: {e}")

st.write(f"📄 Loaded {len(documents)} documents")

# Build vector store
if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        st.error("❌ No text chunks created. Check chunk size or document contents.")
        st.stop()

    # Load OpenAI API key from Streamlit secrets
    api_key = st.secrets["OPENAI_API_KEY"]

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=api_key
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.subheader("💬 Ask a question")
    query = st.text_input("Type your question here:")

    if query:
        with st.spinner("🤖 Thinking..."):
            response = qa_chain.invoke({"query": query})
        st.success("✅ Answer")
        st.write(response)
else:
    st.warning("⚠️ No documents found in the 'data/' folder.")
