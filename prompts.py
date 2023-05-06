from langchain import PromptTemplate

prompt_template = """我提供给你的上下文可能有用，如果没用或者与问题关联性不大，请你自行思考答案

{context}

问题: {question}
有用的回答:"""
HHJS_QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)