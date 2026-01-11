import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM settings."""

    model_name: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "1.0"))
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Specialized settings
    creative_temperature: float = float(os.getenv("LLM_CREATIVE_TEMPERATURE", "1.0"))
    strict_temperature: float = float(os.getenv("LLM_STRICT_TEMPERATURE", "0.0"))


config = LLMConfig()
