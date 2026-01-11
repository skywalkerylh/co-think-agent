from langchain_openai import ChatOpenAI

from src.config import config
from src.tool import generate_ppt


def get_model(temperature: float = None) -> ChatOpenAI:
    """Factory function to create a ChatOpenAI instance with specific configuration."""
    return ChatOpenAI(
        openai_api_key=config.openai_api_key,
        model=config.model_name,
        temperature=temperature if temperature is not None else config.temperature,
    )


# Default model instance (using default config temperature)
model = get_model()

# Specialized models
model_strict = get_model(temperature=config.strict_temperature)
model_creative = get_model(temperature=config.creative_temperature)

tools = [generate_ppt]
model_with_tools = model.bind_tools(tools)
