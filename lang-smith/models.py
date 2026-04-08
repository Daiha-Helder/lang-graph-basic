from dotenv import load_dotenv
import os 
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

_ = load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

_PROVIDER_MAP = {
    'openai':ChatOpenAI,
    'google':ChatGoogleGenerativeAI
}

MODEL_CONFIGS = [
    {
        "key_name": "gemini_2.0_flash",
        "provider": "google",
        "model_name":  "gemini-2.0-flash-lite",
        "temperatura": 1.0
    },
    {
        "key_name": "o4",
        "provider": "openai",
        "model_name": "o4-mini-2025-04-16",
    },
    {
        "key_name": "gpt_4o",
        "provider": "openai",
        "model_name": "gpt-4o-2024-08-06",
    },
    {
        "key_name": "gpt_3_5",
        "provider": "openai",
        "model_name":  "gpt-3.5-turbo",
    },
]

def _create_chat_model(
                model_name: str,
                provider: str,
                temperature: float | None = None
                ):
    
    if provider not in _PROVIDER_MAP:
        raise ValueError(f"Provedor não suportado: {provider}. Provedores suportados são: {list(_PROVIDER_MAP.keys())}")

    model_class = _PROVIDER_MAP[provider]
    params = {"model": model_name}

    if temperature is not None:
        params["temperature"] = temperature

    return model_class(**params)

models = {}

for config in MODEL_CONFIGS:
    models[config["key_name"]] = _create_chat_model(
        model_name=config["model_name"],
        provider=config["provider"],
        temperature=config.get("temperature")
    )