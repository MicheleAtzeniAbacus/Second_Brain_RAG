import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
load_dotenv()


def get_anthropic_client(model: str = "claude-haiku-4-5",
    temperature: float = 0.0,
    ) -> ChatAnthropic:
    """Return a configured OpenAI client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set. Check your .env file.")
    return ChatAnthropic(model_name=model,
    temperature= temperature,
    # max_tokens=,
    # timeout=,
    # max_retries=,
    )
