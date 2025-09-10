from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import UploadFile
from pydantic import BaseModel


class ProviderEnum(str, Enum):
    GEMINI = "GEMINI"
    OPENAI = "OPENAI"
    SARVAM = "SARVAM"


class SuggestRealtimeRequest(BaseModel):
    id: str
    transcript: Optional[str] = None
    transcript_file: Optional[UploadFile] = None
    provider: ProviderEnum = ProviderEnum.GEMINI


class SuggestBatchRequest(BaseModel):
    processing_id: Optional[str] = None
    data: List[SuggestRealtimeRequest]
    webhook_url: Optional[str] = None


class SuggestResponse(BaseModel):
    id: str
    run_id: str
    timestamp: int
    suggestion: Dict[str, Any]
    processing_time: Optional[float] = None


class BatchResponse(BaseModel):
    status: str
    processing_id: str
    timestamp: int
