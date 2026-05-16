import os
# 设置 HuggingFace 镜像（如果还用不到可删除）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_community.chat_models import ChatZhipuAI
# 关键：改用智谱的 Embeddings
from langchain_community.embeddings import ZhipuAIEmbeddings

# ---------- 配置 ----------
# 从环境变量读取 API Key（Streamlit Secrets 中配置）
ZHIPUAI_API_KEY = os.environ.get("ZHIPUAI_API_KEY")

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

# 2. 构建向量库（使用智谱 embedding API，完全无需本地模型）
@st.cache_resource
def build_vectorstore():
    chunks = load_docs()
    if not chunks:
        return None
    # 注意：这里使用智谱的 embedding 接口
    embeddings = ZhipuAIEmbeddings(api_key=ZHIPUAI_API_KEY, model="embedding-2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    vectorstore.persist()
    return vectorstore

# 3. 创建 RAG 链（使用智谱大模型）
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