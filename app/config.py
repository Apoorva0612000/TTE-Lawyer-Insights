import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", 100))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 40))

APM_SERVICE_NAME = os.getenv("ES_SERVICE_NAME_BE", "lawyer-suggestion-api")
APM_SECRET_TOKEN = os.getenv("ES_SECRET_TOKEN", "KDo6rLPpUifgsu2K1y")
APM_SERVER_URL = os.getenv(
    "ES_SERVER_URL",
    "https://f5ecd6a3f21740d9b3e39112390edf9e.apm.ap-south-1.aws.elastic-cloud.com:443",
)
APM_ENVIRONMENT = os.getenv("APM_ENVIRONMENT", "prod")
APM_LOG_LEVEL = os.getenv("APM_LOG_LEVEL", "debug")
APM_DEBUG = os.getenv("APM_DEBUG", "true").lower() == "true"
