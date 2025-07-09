# chatbot.py
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Set up page
st.set_page_config(page_title="Smart Chatbot", layout="wide")
st.title("🤖 Document Chatbot (Optimized for Cost & Speed)")

# Load OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Load files
data_dir = "data"
documents = []
for filename in os.listdir(data_dir):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(data_dir, filename))
    elif filename.endswith(".xlsx"):
        loader = UnstructuredExcelLoader(os.path.join(data_dir, filename))
    else:
        continue
    documents.extend(loader.load())

# Display document status
if not documents:
    st.warning("⚠️ No documents uploaded. Please add files to the `data/` folder.")
    st.stop()
else:
    st.success(f"📄 {len(documents)} document{'s' if len(documents) > 1 else ''} uploaded successfully.")

# Smart splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Build vector DB
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(chunks, embedding_model)
retriever = db.as_retriever()

# Main LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, max_tokens=256, openai_api_key=openai_api_key)

# Setup QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Input
query = st.text_input("🔍 Ask a question")

if query:
    if len(query) > 100:
        st.warning("⚠️ Query too long (max 100 characters allowed). Please shorten your question.")
        st.stop()
    with st.spinner("💡 Thinking..."):
        result = qa_chain.invoke({"query": query})
        raw_answer = result["result"]
        
        # Optional: Summarize if too long
        if len(raw_answer.split()) > 100:  # you can fine-tune this
            summary_chain = load_summarize_chain(llm=llm, chain_type="stuff")
            docs = [Document(page_content=raw_answer)]
            summary = summary_chain.run(docs)
            st.success("📝 Summarized Answer:")
            st.write(summary)
        else:
            st.success("✅ Answer:")
            st.write(raw_answer)

