import os
from typing import Dict

from dotenv import load_dotenv
from elasticapm import async_capture_span

load_dotenv()

TEST_BEARER_TOKEN = os.getenv("TEST_BEARER_TOKEN")  # gitleaks:allow


@async_capture_span("authenticate")
async def authenticate(authorization_header: str) -> Dict[str, str]:
    """
    Bearer token authentication logic (for testing).
    In a production system, this would involve validating against a secure token service.
    """
    expected_token = f"Bearer {TEST_BEARER_TOKEN}"
    if authorization_header != expected_token:
        return {"error": "Invalid authorization header or token"}
    return {"status": "success"}
