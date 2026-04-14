# =============================
# IA GENERATIVA COM RAG PERSISTENTE COM CHROMADB
# =============================

import os
import tempfile
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================
# CONFIGURAÇÃO BASE
# =============================

# 📌 DIRETÓRIO PERSISTENTE
CHROMA_DIR = r"C:\DSA\Fundamentospython\Cap14\chroma_db"

# Embeddings (carregado uma única vez)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/msmarco-bert-base-dot-v5"
)

# Inicializa DB persistente
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# =============================
# STREAMLIT UI
# =============================

st.set_page_config(page_title="Mdje", page_icon=":100:", layout="wide")

with st.sidebar:
    st.header("Configurações")
    api_key = st.text_input("Coloque sua GROQ API Key", type="password")

st.title("✈️ Assistente RAG Persistente")
st.caption("Base de conhecimento contínua com ChromaDB")

if not api_key:
    st.warning("Informe a API Key")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_tokens=1024
)

# =============================
# FUNÇÃO DE INGESTÃO
# =============================

def adicionar_pdf(pdf_bytes):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    docs = PyPDFLoader(tmp_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    # Adiciona metadados
    for chunk in chunks:
        chunk.metadata["source"] = tmp_path
        chunk.metadata["domain"] = "aduana"

    vectordb.add_documents(chunks)
    vectordb.persist()

# =============================
# UPLOAD
# =============================

pdf_file = st.file_uploader("Adicionar PDF à base", type=["pdf"])

if pdf_file:
    with st.spinner("Processando e armazenando..."):
        adicionar_pdf(pdf_file.read())
        st.success("Documento adicionado à base!")

# =============================
# RETRIEVER
# =============================

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# =============================
# PROMPT
# =============================

system_block = """
Você é um assistente especializado em aviação comercial, logística internacional, despacho aduaneiro e transitários.

Use APENAS o contexto fornecido.
Se a resposta não estiver no contexto, diga claramente.

Estrutura:
1. Contexto
2. Pontos principais
3. Riscos
4. Próximos passos
"""

qa_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_block),
    ("human", "Pergunta: {question}\n\nContexto:\n{context}")
])

# =============================
# FORMATAÇÃO
# =============================

def formatar_docs(docs):
    textos = []
    for d in docs:
        textos.append(d.page_content[:800])
    return "\n\n".join(textos)

# =============================
# INPUT USUÁRIO
# =============================

pergunta = st.text_area("Faça sua pergunta")

if st.button("Perguntar"):

    if not pergunta:
        st.warning("Digite uma pergunta")
        st.stop()

    pipeline = RunnableParallel(
        context=retriever | formatar_docs,
        question=RunnablePassthrough()
    ) | qa_prompt | llm | StrOutputParser()

    with st.spinner("Consultando base..."):
        resposta = pipeline.invoke(pergunta)

    st.markdown("### Resposta")
    st.write(resposta)

# =============================
# FIM
# =============================
