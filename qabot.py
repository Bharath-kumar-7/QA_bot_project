# ---------------- IMPORTS ----------------
import os
import warnings
import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings

warnings.filterwarnings("ignore")


# ---------------- LLM (Gemini) ----------------
def get_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    return llm


# ---------------- DOCUMENT LOADER ----------------
def document_loader(file):
    loader = PyPDFLoader(file.name)
    documents = loader.load()
    return documents


# ---------------- TEXT SPLITTER ----------------
def text_splitter(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks


# ---------------- EMBEDDINGS (Gemini) ----------------
def embedding_model():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings


# ---------------- VECTOR DATABASE ----------------
def vector_database(chunks):
    embeddings = embedding_model()
    vectordb = Chroma.from_documents(chunks, embeddings)
    return vectordb


# ---------------- RETRIEVER ----------------
def retriever(file):
    documents = document_loader(file)
    chunks = text_splitter(documents)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()


# ---------------- QA CHAIN ----------------
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
    )

    response = qa_chain.invoke(query)
    return response["result"]


# ---------------- GRADIO UI ----------------
rag_app = gr.Interface(
    fn=retriever_qa,
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Ask Question")],
    outputs=gr.Textbox(label="Answer"),
    title="RAG QA Bot (Gemini Version)",
    description="Upload a PDF and ask questions from it.",
)

if __name__ == "__main__":
    rag_app.launch(server_port=7860)
