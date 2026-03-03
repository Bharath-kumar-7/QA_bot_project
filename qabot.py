# ---------------- IMPORTS ----------------
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr
import warnings

warnings.filterwarnings("ignore")


# ---------------- LLM ----------------
def get_llm():
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }

    llm = WatsonxLLM(
        model_id="mistralai/mistral-medium-2505",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=parameters,
    )

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


# ---------------- EMBEDDING MODEL ----------------
def watsonx_embedding():

    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }

    embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )

    return embedding


# ---------------- VECTOR DATABASE ----------------
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
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
    title="PDF Question Answering Bot",
    description="Upload a PDF and ask questions from it.",
)


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    rag_app.launch(server_port=7860)
