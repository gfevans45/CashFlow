import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_CONFIG_CACHE = None


def load_config(config_path: str = None) -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "settings.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _CONFIG_CACHE = config
    return config


def get_alpaca_credentials() -> dict:
    return {
        "api_key": os.getenv("ALPACA_API_KEY", ""),
        "secret_key": os.getenv("ALPACA_SECRET_KEY", ""),
        "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
    }


def get_kalshi_credentials() -> dict:
    return {
        "email": os.getenv("KALSHI_EMAIL", ""),
        "password": os.getenv("KALSHI_PASSWORD", ""),
        "base_url": os.getenv("KALSHI_BASE_URL", "https://demo-api.kalshi.co"),
    }
