# langchain_rag
a test model of langchain_rag
RAG 分为两个主要部分：检索（Retrieval）和生成（Generation）。在回答一个问题时，RAG 首先从一个大型的文档集合中检索相关信息。这个检索过程通常是基于问题的内容，旨在找到能够帮助生成更准确答案的文档。检索到的文档接着被送入一个生成模型，如 GPT 系列模型，这个模型会结合检索到的内容和它自身的知识来生成答案。
