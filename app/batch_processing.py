import asyncio
import uuid
from typing import List, Optional

# NEW: Import APM
from elasticapm import async_capture_span

from app.feedback_analysis import suggestion_caller_function
from app.logging_config import logger
from app.models import SuggestRealtimeRequest
from app.utils import send_webhook_notification


@async_capture_span("process_batch_async")  # NEW: APM span
async def process_batch_async(
    processing_id: str, data: List[SuggestRealtimeRequest], webhook_url: Optional[str]
):
    """Process batch of transcription requests asynchronously."""
    logger.info(f"[{processing_id}] Batch processing started with {len(data)} requests")
    semaphore = asyncio.Semaphore(5)

    @async_capture_span("process_single_request")
    async def process_single_request(request_data):
        async with semaphore:
            run_id = uuid.uuid4().hex
            try:
                result = await suggestion_caller_function(
                    processing_id, request_data, "BATCH", run_id
                )
                return result
            except Exception as e:
                logger.error(
                    f"[{processing_id} {run_id}] Error processing request: {str(e)}"
                )
                return {"error": str(e), "id": request_data.id, "run_id": run_id}

    tasks = [process_single_request(req) for req in data]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    if webhook_url:
        await send_webhook_notification(webhook_url, processing_id, results)
    logger.info(f"[{processing_id}] Batch processing completed")
