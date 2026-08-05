import os
from functools import lru_cache

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()


@lru_cache
def get_sarvam_client() -> SarvamAI:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set. Add it to your .env file.")
    return SarvamAI(api_subscription_key=api_key)
