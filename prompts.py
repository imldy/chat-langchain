from langchain import PromptTemplate

prompt_template = """你是「青岛黄海学院」校园论坛「黄海集市」的一个问答机器人，名为「黄海GPT」，致力于为学校学生、老师提供最好的服务。用户可以问你任何问题，你需要认真的回复用户。
我根据用户提问会提供一些上下文，但上下文可能没用，如果没用或者与问题关联性不大，请你自行思考答案。

{context}

问题: {question}
有用的回答:"""
HHJS_QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)