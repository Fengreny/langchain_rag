from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import SelfQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
import openai

# 加载.env文件和设置环境变量
from dotenv import load_dotenv
load_dotenv()  # 先加载.env文件

openai.api_key = os.getenv("OPENAI_API_KEY")

# 加载文档
pdf_file = "D:/chatGPT调研报告.pdf"
loader = PyPDFLoader(pdf_file)
docs = loader.load()

# 文档分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_documents(docs)

# 向量库存储
embedding = OpenAIEmbeddings()
persist_directory = 'docs/chroma/'
vectordb = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()

# 读取向量数据库
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# 检索
metadata_field_info = [
    AttributeInfo(name="source", description="The lecture the chunk is from, should be `D:/chatGPT调研报告.pdf`", type="string"),
    AttributeInfo(name="page", description="The page from the lecture", type="integer"),
]
document_content_description = "Lecture notes"
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
self_query_retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    search_type="mmr",
    search_kwargs={'k': 5, 'fetch_k': 10},
    verbose=True
)

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=self_query_retriever)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=compression_retriever, memory=memory)

while True:
    question = input("请输入您的问题（输入'退出'来结束）: ")
    if question.lower() == '退出':
        break
    result = qa({"question": question})
    print("回答:", result)