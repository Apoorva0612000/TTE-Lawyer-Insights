import asyncio
import logging
import time
import uuid
import warnings
from urllib.parse import urlparse

import requests
from elasticapm import async_capture_span
from elasticapm.contrib.starlette import ElasticAPM, make_apm_client
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.auth import authenticate
from app.batch_processing import process_batch_async
from app.config import (
    APM_ENVIRONMENT,
    APM_SECRET_TOKEN,
    APM_SERVER_URL,
    APM_SERVICE_NAME,
)
from app.feedback_analysis import suggestion_caller_function
from app.logging_config import logger
from app.models import (
    BatchResponse,
    SuggestBatchRequest,
    SuggestRealtimeRequest,
    SuggestResponse,
)
from app.utils import check_rate_limit

# Suppress the SSL warnings
warnings.filterwarnings(
    "ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning
)


def init_apm():
    """Initialize APM client with fixed configuration."""
    try:
        # Configure logging for Elastic APM - REDUCED LOGGING
        apm_logger = logging.getLogger("elasticapm")
        # Change from DEBUG to INFO to reduce verbosity
        apm_logger.setLevel(logging.INFO)

        # More streamlined configuration
        apm_config = {
            "SERVICE_NAME": APM_SERVICE_NAME,
            "SECRET_TOKEN": APM_SECRET_TOKEN,
            "SERVER_URL": APM_SERVER_URL,
            "ENVIRONMENT": APM_ENVIRONMENT,
            # Critical settings
            "ENABLED": True,
            "RECORDING": True,
            "DISABLE_SEND": False,
            # Reduced logging
            "DEBUG": False,  # Changed from True to reduce logging
            "LOG_LEVEL": "trace",  # Changed from debug
            # Performance settings
            "METRICS_INTERVAL": "30s",
            "TRANSACTION_SAMPLE_RATE": 1.0,
            "CENTRAL_CONFIG": True,
            "VERIFY_SERVER_CERT": False,
            "SERVER_TIMEOUT": "10s",
            # Only capture errors, not all bodies
            "CAPTURE_BODY": "errors",
            "STACK_TRACE_LIMIT": 50,
        }

        logger.info("Initializing Elastic APM client...")
        apm = make_apm_client(apm_config)

        # Initialize instrumentation
        from elasticapm import instrument

        instrument()
        logger.info("APM instrumentation applied")

        # Create a test transaction
        apm.begin_transaction("startup")
        time.sleep(0.1)
        apm.end_transaction("startup", 200)

        return apm

    except Exception as e:
        logger.error(f"APM initialization failed: {str(e)}", exc_info=True)
        return None


# Initialize APM client
apm = init_apm()


app = FastAPI(title="AI suggestion API", version="1.0.0")

# CORS setup (customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if apm:
    try:
        app.add_middleware(ElasticAPM, client=apm)
        logger.info("Elastic APM middleware added successfully")
    except Exception as e:
        logger.error(f"Failed to add APM middleware: {str(e)}", exc_info=True)
        apm = None


@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        # Skip auth for health, diagnostic and test endpoints
        if request.url.path in [
            "/health",
            "/test-apm",
            "/apm-diagnostics",
            "/test-apm-connection",
            "/force-apm-transaction",
        ]:
            response = await call_next(request)
            return response

        # Get authorization header
        authorization = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header required")

        # Authenticate
        auth_result = await authenticate(authorization)
        if "error" in auth_result:
            raise HTTPException(status_code=401, detail=auth_result["error"])

        response = await call_next(request)
        return response
    except HTTPException as exc:
        logger.error(f"HTTPException: {exc.detail}")
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    except Exception as exc:
        logger.error(f"Unhandled Exception: {str(exc)}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.get("/health")
@async_capture_span("health")
async def health():
    """Health check endpoint"""
    if apm:
        # Force transaction to appear in APM
        apm.begin_transaction("health_check")
        time.sleep(0.05)  # Small delay
        apm.end_transaction("health_check", 200)

    return {"status": "ok", "apm_enabled": apm is not None}


async def get_transaction_name(request: Request):
    """Get a descriptive transaction name from the request path"""
    path = request.url.path

    # Special case for API endpoints with IDs
    if path.startswith("/api/"):
        # Extract the base endpoint without IDs
        parts = path.split("/")
        if len(parts) > 2:
            return f"{parts[1]}_{parts[2]}"  # e.g. "api_transcribe-realtime"

    # Use the path without the leading slash
    return path[1:] if path.startswith("/") else path


@app.middleware("http")
async def apm_transaction_middleware(request: Request, call_next):
    """Custom middleware to ensure each request has a transaction"""
    # Skip if APM is not initialized
    if not apm:
        return await call_next(request)

    # Get a meaningful transaction name
    transaction_name = await get_transaction_name(request)

    # Start transaction explicitly
    apm.begin_transaction("request")

    try:
        # Execute the request
        response = await call_next(request)

        # End transaction with success
        apm.end_transaction(transaction_name, response.status_code)
        return response
    except Exception as e:  # noqa: F841
        # Capture exception and end transaction with error
        apm.capture_exception()
        apm.end_transaction(transaction_name, 500)
        raise


@app.get("/test-apm")
async def test_apm():
    """Test APM functionality"""
    if apm:
        logger.info("APM test endpoint called")

        # Create a transaction manually to ensure it appears
        apm.begin_transaction("test_transaction")

        # Add some spans
        with apm.capture_span("custom_test_span"):
            time.sleep(0.1)  # Simulate some work

        # End transaction explicitly
        apm.end_transaction("test_transaction", 200)

        return {
            "status": "APM test transaction created",
            "apm_enabled": True,
            "check_kibana": "Please check Kibana APM UI to see if this transaction appears",
            "timestamp": int(time.time()),
        }
    else:
        return {"status": "APM not initialized", "apm_enabled": False}


@app.get("/test-apm-connection")
async def test_apm_connection():
    """Test direct connection to APM server"""
    apm_url = APM_SERVER_URL
    parsed_url = urlparse(apm_url)

    # Test connection by hitting the APM server info endpoint
    test_url = f"{parsed_url.scheme}://{parsed_url.netloc}/info"

    try:
        response = requests.get(
            test_url,
            headers={"Authorization": f"Bearer {APM_SECRET_TOKEN}"},
            timeout=5,
            verify=False,  # Match your VERIFY_SERVER_CERT setting
        )
        return {
            "status": response.status_code,
            "connection_successful": response.status_code == 200,
            "url_tested": test_url,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "url_tested": test_url}


@app.get("/force-apm-transaction")
async def force_apm_transaction():
    """Force create an APM transaction with nested spans"""
    if not apm:
        return {"status": "APM not initialized"}

    # Create a transaction explicitly
    apm.begin_transaction("forced_transaction")

    # Add some spans
    with apm.capture_span("span1", span_type="custom"):
        time.sleep(0.1)
        with apm.capture_span("span2", span_type="db.query"):
            time.sleep(0.1)

    # End transaction explicitly
    apm.end_transaction("forced_transaction", 200)

    return {
        "status": "Transaction created and sent to APM server",
        "check_kibana": "Please check your APM UI in Kibana for a transaction named 'forced_transaction'",
    }


@app.get("/apm-diagnostics")
async def apm_diagnostics():
    """Provide detailed APM diagnostics and test error reporting"""
    if not apm:
        return {"status": "APM not initialized"}

    # Check client configuration
    config_info = {
        "service_name": apm.config.service_name,
        "environment": apm.config.environment,
        "server_url": apm.config.server_url,
        "enabled": apm.config.enabled,
        "recording": apm.config.recording,
    }

    # Create a test transaction
    apm.begin_transaction("diagnostics")

    # Test error capture
    try:
        # Deliberately cause an error
        result = 1 / 0
        logger.info(f"Result: {result}")
    except Exception as e:  # noqa: F841
        apm.capture_exception()

    # End transaction
    apm.end_transaction("diagnostics", 200)

    return {
        "status": "Diagnostics complete",
        "config": config_info,
        "test_error_sent": True,
        "check_kibana": "Please check your APM UI in Kibana for an error (division by zero) in a transaction named 'diagnostics'",
    }


@app.post("/api/suggest-realtime", response_model=SuggestResponse)
async def transcribe_realtime(
    request: SuggestRealtimeRequest,
    authorization: str = Header(..., alias="Authorization"),
    req: Request = None,
):
    # 1. Authentication
    auth_result = await authenticate(authorization)
    if "error" in auth_result:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # 2. Rate limiting
    await check_rate_limit(req)
    # 3. Progress tracking stub
    run_id = uuid.uuid4().hex
    logger.info(f"[{run_id} {request.id}] [PROGRESS] Start")
    # 4. Process transcription
    result = await suggestion_caller_function(None, request, "REALTIME", run_id, None)
    logger.info(f"[{run_id} {request.id}] [PROGRESS] Done")
    return result


@app.post("/api/suggest-batch", response_model=BatchResponse)
async def transcribe_batch(
    request: SuggestBatchRequest,
    authorization: str = Header(..., alias="Authorization"),
    req: Request = None,
):
    # 1. Authentication
    auth_result = await authenticate(authorization)
    if "error" in auth_result:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # 2. Rate limiting
    await check_rate_limit(req)
    # 3. Validate batch size
    if len(request.data) > 40:
        raise HTTPException(status_code=400, detail="Max 40 requests in a batch")
    # 4. Progress tracking stub
    processing_id = request.processing_id or f"P{uuid.uuid4().hex}"
    logger.info(f"[{processing_id}] [PROGRESS] Batch start")
    # 5. Start async batch processing
    asyncio.create_task(
        process_batch_async(processing_id, request.data, request.webhook_url)
    )
    logger.info(f"[{processing_id}] [PROGRESS] Batch submitted")
    # 6. Return immediate response
    return BatchResponse(
        status="success", processing_id=processing_id, timestamp=int(time.time())
    )
