import os
import warnings
import gradio as gr
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

warnings.filterwarnings("ignore")

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")


# ---------------- LLM ----------------
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        google_api_key=GOOGLE_API_KEY,
    )


# ---------------- Document Loader ----------------
def document_loader(file):
    loader = PyPDFLoader(file.name)
    return loader.load()


# ---------------- Text Splitter ----------------
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    return splitter.split_documents(data)


# ---------------- Embeddings ----------------
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )


# ---------------- Vector Database ----------------
def vector_database(chunks):
    embeddings = get_embeddings()
    return Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")


# ---------------- Retriever ----------------
def retriever(file):
    docs = document_loader(file)
    chunks = text_splitter(docs)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()


# ---------------- QA Chain ----------------
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
    )

    response = qa.invoke(query)
    return response["result"]


# ---------------- Gradio Interface ----------------
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_types=[".pdf"]),
        gr.Textbox(label="Ask Question", lines=2),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Gemini RAG Chatbot",
    description="Upload a PDF and ask questions about its content.",
)

if __name__ == "__main__":
    rag_application.launch(share=True)
