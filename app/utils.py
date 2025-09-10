from collections import defaultdict
from datetime import datetime
from time import time as now
from typing import Any, Dict, List

import httpx
from elasticapm import async_capture_span
from fastapi import HTTPException, Request

from app.logging_config import logger


@async_capture_span("send_webhook_notification")
async def send_webhook_notification(
    webhook_url: str, processing_id: str, results: List[Dict[str, Any]]
):
    """
    Send webhook notification asynchronously to the given webhook_url.
    Payload format matches the expected webhook schema:
    {
        "processing_id": str,
        "status": "completed" | "failed",
        "results": [
            {
                "id": str,
                "run_id": str,
                "result": {
                    "suggestion": {
                        "strengths": str,
                        "clarity": str,
                        "tone": str,
                        "depth": str
                    }
                },
                "timestamp": str (ISO format)
            }
        ],
        "errors": []
    }
    """
    # Convert results to the expected format
    formatted_results = []
    errors = []

    for result in results:
        try:
            # Handle different result types (Pydantic models, dicts, or error objects)
            if hasattr(result, "dict"):  # Pydantic model
                result_dict = result.dict()
            elif isinstance(result, dict):
                result_dict = result
                logger.info()
            else:
                # Convert to dict manually
                result_dict = {
                    "id": getattr(result, "id", "unknown"),
                    "run_id": getattr(result, "run_id", "unknown"),
                    "timestamp": getattr(result, "timestamp", int(now())),
                    "suggestion": getattr(result, "suggestions", {}),
                    "processing_time": getattr(result, "processing_time", None),
                    "error": getattr(result, "error", None),
                }

            # Check if this result has an error
            if "error" in result_dict and result_dict["error"]:
                errors.append(
                    {
                        "id": result_dict.get("id", "unknown"),
                        "run_id": result_dict.get("run_id", "unknown"),
                        "error": result_dict["error"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            else:
                # Format successful result according to webhook schema
                formatted_result = {
                    "id": result_dict.get("id", "unknown"),
                    "run_id": result_dict.get("run_id", "unknown"),
                    "suggestion": result_dict.get("suggestion", {}),
                    "timestamp": datetime.now().isoformat(),
                }

                # NEW: Add full transcript if available

                formatted_results.append(formatted_result)

        except Exception as e:
            logger.error(f"Failed to format result object: {e}")
            errors.append(
                {
                    "id": "unknown",
                    "run_id": "unknown",
                    "error": f"Failed to format result: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            )

    # Determine overall status
    status = "completed" if len(errors) == 0 else "failed"

    # Create final payload matching webhook schema
    payload = {
        "processing_id": processing_id,
        "status": status,
        "results": formatted_results,
        "errors": errors,
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(webhook_url, json=payload, timeout=30)
            resp.raise_for_status()
        logger.info(
            f"Webhook sent successfully to {webhook_url} for processing_id {processing_id}"
        )
        logger.info(
            f"Webhook payload: {len(formatted_results)} results, {len(errors)} errors, status: {status}"
        )
    except Exception as e:
        logger.error(
            f"Failed to send webhook to {webhook_url} for processing_id {processing_id}: {e}"
        )


# In-memory rate limiting (per-IP)
rate_limit_cache = defaultdict(list)
RATE_LIMIT = 100  # requests per minute


@async_capture_span("check_rate_limit")
async def check_rate_limit(request: Request):
    """Check and enforce rate limit per IP (100/min). Raise HTTPException 429 if exceeded."""
    ip = request.client.host if request.client else "unknown"
    now_ts = now()
    window = 60  # seconds
    timestamps = rate_limit_cache[ip]
    # Remove timestamps older than 60 seconds
    rate_limit_cache[ip] = [t for t in timestamps if now_ts - t < window]
    if len(rate_limit_cache[ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Max 100 requests per minute."
        )
    rate_limit_cache[ip].append(now_ts)
