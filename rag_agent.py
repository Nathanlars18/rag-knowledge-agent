import os
# 设置 HuggingFace 镜像（放在所有其他导入之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
# 1. 使用 langchain_community 下的文档加载器（已修复）
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# 2. 使用 langchain_text_splitters（已修复）
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 3. 使用 langchain_community 下的 Embeddings 和 Vectorstores（已修复）
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# 4. 使用 langchain_classic 下的 RetrievalQA（最新修复）
from langchain_classic.chains import RetrievalQA
from langchain_community.chat_models import ChatZhipuAI

# ---------- 配置 ----------
# 请替换成你自己的智谱 API Key
ZHIPUAI_API_KEY = "b0c1ad3bb00444c0a8552cd580c164dc.pDy2ePa1NtZtAaq0"

# 1. 加载并切分文档
def load_docs(directory="./docs"):
    docs = []
    if not os.path.exists(directory):
        st.warning(f"文件夹 {directory} 不存在，请先创建并放入文档")
        return []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(path, encoding="utf-8")
        else:
            continue
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# 2. 构建向量库（使用本地 HuggingFace embedding，免费）
@st.cache_resource
def build_vectorstore():
    chunks = load_docs()
    if not chunks:
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    vectorstore.persist()
    return vectorstore

# 3. 创建 RAG 链（使用智谱AI）
def get_qa_chain(vectorstore):
    llm = ChatZhipuAI(
        api_key=ZHIPUAI_API_KEY,
        model="glm-4",
        temperature=0.1,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

# 4. Streamlit 界面
st.set_page_config(page_title="个人知识库助手", layout="wide")
st.title("📚 你的 RAG 知识问答 Agent")
st.markdown("把文档放进 `docs/` 文件夹，然后问任何关于这些文档的问题。")

with st.spinner("正在加载知识库（首次运行会稍慢）..."):
    vectordb = build_vectorstore()
    if vectordb is None:
        st.error("请先在项目根目录下创建 docs/ 文件夹，并放入至少一个 .txt / .pdf / .md 文件")
        st.stop()
    qa_chain = get_qa_chain(vectordb)

query = st.text_input("💬 输入你的问题：")
if query:
    with st.spinner("Agent 正在思考（检索 → 推理 → 生成引用）..."):
        result = qa_chain({"query": query})
        st.markdown("### 回答")
        st.write(result["result"])
        with st.expander("📄 查看引用来源"):
            for i, doc in enumerate(result["source_documents"]):
                st.write(f"**来源 {i+1}:** {doc.metadata.get('source', '未知')}")
                st.write(doc.page_content[:300] + "...")