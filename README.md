# langchain_rag
a test model of langchain_rag
RAG 分为两个主要部分：检索（Retrieval）和生成（Generation）。在回答一个问题时，RAG 首先从一个大型的文档集合中检索相关信息。这个检索过程通常是基于问题的内容，旨在找到能够帮助生成更准确答案的文档。检索到的文档接着被送入一个生成模型，如 GPT 系列模型，这个模型会结合检索到的内容和它自身的知识来生成答案。


检索信息（RAG的“检索”部分）:

使用了一个假设的 retrieve_information 函数来模拟从外部数据源获取信息的过程。在实际应用中，这可以是一个查询数据库、调用外部API或执行其他类型的搜索操作的函数。
这个函数接收用户的查询作为输入，并返回与该查询相关的信息。
生成回应（RAG的“生成”部分）:

使用 LangChain 中的 LLMChain 和 PromptTemplate 来生成文本回应。
PromptTemplate 定义了生成回应时使用的文本模板。原先包含 retrieved_information 作为输入变量，但后来调整为仅依赖 human_input。
LLMChain 则负责处理实际的生成过程。它接收整合了用户输入和检索到的信息的文本，并基于此生成回应。
对话管理:

使用了 LangChain 的 ConversationBufferWindowMemory 来管理和存储对话历史，这对于保持对话的连贯性是有用的。
用户的每个新输入和由模型生成的回应都被添加到这个对话历史中。
主程序:

在主程序中，用户的输入首先被用于检索相关信息，然后这些信息被合并到用户的原始输入中，形成 combined_input。
这个 combined_input 然后被作为输入传递给 LLMChain 来生成回应。
这个过程在一个循环中进行，直到用户输入“退出”为止。
