import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

# 假设的检索函数
def retrieve_information(query):
    # 这里应该包含实际的检索逻辑
    # 为了示例，我们假设这个函数返回与查询相关的信息
    return f"检索到的信息：关于 '{query}' 的相关历史背景和数据。"

# 从环境变量获取 OpenAI API 密钥
openai_api_key = os.environ.get('OPENAI_API_KEY')
if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=openai_api_key)

# 创建 PromptTemplate
# 修改 PromptTemplate
template = """Assistant is a large language model trained by OpenAI.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "human_input"],  # 移除 'retrieved_information'
    template=template
)


chatgpt_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=5),
)

# 主程序
def main():
    print("开始对话（输入'退出'结束对话）:")
    while True:
        user_input = input("你: ")
        if user_input.lower() == '退出':
            break

        # 检索阶段
        retrieved_info = retrieve_information(user_input)

        # 将检索到的信息和用户输入合并
        combined_input = f"{retrieved_info}\n\n{user_input}"

        # 调用 LangChain LLMChain 来生成答案
        output = chatgpt_chain.predict(
            history=chatgpt_chain.memory.buffer,
            human_input=combined_input  # 传递合并后的输入
        )
        print(f"AI: {output}")

if __name__ == "__main__":
    main()
