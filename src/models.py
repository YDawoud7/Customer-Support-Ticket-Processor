import os

from langchain_core.language_models import BaseChatModel
from typing_extensions import TypedDict

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class ModelConfig(TypedDict, total=False):
    """Maps each graph node to a model name string.

    Model names are auto-detected by prefix:
      - ``deepseek-*`` → ChatOpenAI pointed at the DeepSeek API
      - anything else  → ChatAnthropic (default)
    """

    classifier: str
    billing: str
    technical: str
    general: str
    quality_check: str


def create_chat_model(model_name: str) -> BaseChatModel:
    """Instantiate a chat model based on the model name prefix."""
    if model_name.startswith("deepseek"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            base_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model_name)


ALL_CLAUDE: ModelConfig = {
    "classifier": DEFAULT_MODEL,
    "billing": DEFAULT_MODEL,
    "technical": DEFAULT_MODEL,
    "general": DEFAULT_MODEL,
    "quality_check": DEFAULT_MODEL,
}

COST_OPTIMIZED: ModelConfig = {
    "classifier": "deepseek-chat",
    "billing": DEFAULT_MODEL,
    "technical": DEFAULT_MODEL,
    "general": DEFAULT_MODEL,
    "quality_check": "deepseek-chat",
}

ALL_DEEPSEEK: ModelConfig = {
    "classifier": "deepseek-chat",
    "billing": "deepseek-chat",
    "technical": "deepseek-chat",
    "general": "deepseek-chat",
    "quality_check": "deepseek-chat",
}

PRESET_CONFIGS: dict[str, ModelConfig] = {
    "all_claude": ALL_CLAUDE,
    "cost_optimized": COST_OPTIMIZED,
    "all_deepseek": ALL_DEEPSEEK,
}
