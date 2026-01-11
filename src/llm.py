import os

from langchain_openai import ChatOpenAI

from src.tool import generate_ppt

model = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=1,
)
tools = [generate_ppt]
model_with_tools = model.bind_tools(tools)