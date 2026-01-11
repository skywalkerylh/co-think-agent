from langchain_google_genai import ChatGoogleGenerativeAI
from src.settings import settings

llm = ChatGoogleGenerativeAI(
    google_api_key=settings.GOOGLE_API_KEY,
    model=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE,
)
