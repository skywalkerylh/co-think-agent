import os
from dotenv import load_dotenv

load_dotenv()

class Settings():

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    MODEL_NAME = "gemini-2.5-flash"
    
    TEMPERATURE = 1

    MAX_RETRIES = 3


settings = Settings()  