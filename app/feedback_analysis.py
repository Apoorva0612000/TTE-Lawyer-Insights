import asyncio
import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from cachetools import TTLCache

# from elasticapm import async_capture_span
from fastapi import HTTPException, UploadFile
from openai import OpenAI
from pinecone import Pinecone

from app.config import GEMINI_API_KEY, OPENAI_API_KEY, PINECONE_API_KEY
from app.logging_config import logger
from app.models import SuggestRealtimeRequest, SuggestResponse

# SETUP
# Load environment variables
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "call-transcripts-index"
# Connect to the index
index = pc.Index(index_name)

# Initialize OpenAI client
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
# Create OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

CHUNK_SIZE_GEMINI = 5000
OVERLAP_GEMINI = 300
DEFAULT_SEGMENT_DURATION_GEMINI = (
    5  # seconds per snippet if model's timestamps are unreliable
)

CHUNK_SIZE_OPENAI = 5000  # larger chunk so model sees more context
OVERLAP_OPENAI = 300  # minimal overlap to reduce duplication
DEFAULT_SEGMENT_DURATION_OPENAI = 5

MAX_OPENAI_CONCURRENT = 5
MAX_GEMINI_CONCURRENT = 5

openai_semaphore = asyncio.Semaphore(MAX_OPENAI_CONCURRENT)
gemini_semaphore = asyncio.Semaphore(MAX_GEMINI_CONCURRENT)


async def process_transcript_input(request_data):
    transcript_content = None
    # Case 1: transcript comes as JSON string or dict
    if request_data.transcript:
        if isinstance(request_data.transcript, str):
            transcript_content = request_data.transcript

    # Case 2: transcript comes as file
    elif getattr(request_data, "transcript_file", None):
        file = request_data.transcript_file
        content = await file.read()
        decoded = content.decode()

        if file.filename.endswith(".txt"):
            transcript_content = decoded
        else:
            raise HTTPException(
                status_code=400, detail="Unsupported file type for transcript"
            )

    else:
        raise HTTPException(status_code=400, detail="No transcript provided")

    return transcript_content


# @async_capture_span("suggestion_caller_function")
async def suggestion_caller_function(
    processing_id: Optional[str],
    request_data: SuggestRealtimeRequest,
    type: str,
    run_id: str,
    file: Optional[UploadFile] = None,
) -> SuggestResponse:
    """Main caller function for transcription processing."""
    start_time = time.time()
    transcript_text = await process_transcript_input(request_data)
    try:
        # Start suggestion process
        logger.info(f"[{run_id} {request_data.id}] Suggestion processing started")
        result = await async_suggestion_with_limit(
            transcript=transcript_text,
            run_id=run_id,
            request_id=request_data.id,
            provider=request_data.provider,
        )

        processing_time = time.time() - start_time

        # Create response with conditional full_transcript
        response_data = {
            "id": request_data.id,
            "run_id": run_id,
            "timestamp": int(time.time()),
            "suggestion": result.get("suggestions", {}),
            "processing_time": processing_time,
        }

        response = SuggestResponse(**response_data)
        logger.info(f"[{run_id} {request_data.id}] Suggestion completed successfully")
        return response
    except Exception as e:
        logger.error(f"[{run_id} {request_data.id}] Suggestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {str(e)}")


async def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding


async def get_embedding_batch(texts: list[str]) -> list[list[float]]:
    """
    Compute embeddings for multiple texts at once using OpenAI API.
    """
    # Remove empty strings
    texts = [t for t in texts if t.strip()]
    if not texts:
        return []

    loop = asyncio.get_event_loop()

    def sync_call():
        response = client.embeddings.create(
            model="text-embedding-3-small", input=texts  # must be a list of strings
        )
        return [item.embedding for item in response.data]

    embeddings = await loop.run_in_executor(None, sync_call)
    return embeddings


def seconds_to_mmss(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


# GEMINI SEGMENTATION

gemini_prompt_cache = TTLCache(maxsize=5000, ttl=6 * 3600)


# @async_capture_span("segregate_with_gemini")
async def segregate_with_gemini(
    full_transcript: str, run_id: str, request_id: str
) -> Dict:
    """Segregate transcript using Google Gemini API"""
    # Dynamic API key check to ensure it's loaded fresh
    if not GEMINI_API_KEY:
        # logger.error(f"[{run_id} {request_id}] Gemini API key not found for summarization")
        return {"error": "Gemini API key not found in environment variables"}

    # logger.info(f"[{run_id} {request_id}] Sending transcript to Gemini for summarization. Transcript length: {len(full_transcript)}")
    chunks = []
    i, chunk_id = 0, 1
    while i < len(full_transcript):
        chunk = full_transcript[i : i + CHUNK_SIZE_GEMINI]
        chunks.append((chunk_id, chunk))
        i += CHUNK_SIZE_GEMINI - OVERLAP_GEMINI
        chunk_id += 1

    logger.info(f"[{run_id} {request_id}] Transcript split into {len(chunks)} chunks.")
    # Prepare the request payload for summarization

    async def process_chunk(chunk_text: str, cid: int):

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"""
                                You are an intelligent legal call transcript assistant. Your task is to segment a lawyer-client call into **multiple continuous segments**, each assigned to one of the following four categories, and output structured JSON.

                                Segment categories:
                                1. Introduction_Greeting
                                2. Client_Problem_Questions
                                3. Legal_Advice_Options
                                4. Next_Steps_Closing

                                Your job includes:
                                - Each segment should have one of the four segment_ids above. Do not create new segment_ids.
                                - Segments must be continuous and non-overlapping.
                                - Use MM:SS format for timestamps.
                                - Identify speakers as "lawyer" or "client".
                                - Return ONLY valid JSON; no explanations or markdown.
                                - Include empty segments if needed.
                                - Create separate segments for each speaker snippet; do NOT combine multiple snippets.
                                - Maintain the order of conversation, keep segments continuous, and do not overlap.
                                - Use best-guess judgment if a snippet is ambiguous, but do not merge it with other snippets.
                                - You may create multiple segments for the same category as needed.
                                - Use best-guess judgment if unsure where text fits.

                                Segment details:

                                1. Introduction_Greeting
                                - Lawyer greetings, introductions, small talk

                                2. Client_Problem_Questions
                                - Client statements describing problem, context, dates, documents
                                - Initial context questions to understand the client’s problem
                                - Lawyer clarifying questions
                                - Exclude advice or solutions

                                3. Legal_Advice_Options
                                - Lawyer-provided advice, recommendations, or legal options
                                - Include pros/cons, analysis of different approaches
                                - Exclude greetings, client problem description, or closing statements

                                4. Next_Steps_Closing
                                - Follow-up instructions, deadlines, documents required, confirmations, closing remarks
                                - Any final questions from client or lawyer




                                Format the response exactly as:

                                {{
                                    "segments": [
                                        {{
                                            "segment_id": "",
                                            "start_timestamp": "",
                                            "end_timestamp": "",
                                            "speaker": "",
                                            "text": ""
                                        }}
                                    ]
                                }}

                                Transcript chunk(part {cid}):
                                {chunk_text}
                            """
                        }
                    ]
                }
            ]
        }
        # Set headers and URL
        headers = {"Content-Type": "application/json"}

        url = f"{GEMINI_GENERATE_URL}?key={GEMINI_API_KEY}"

        prompt_str = payload["contents"][0]["parts"][0]["text"]
        prompt_key = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()  # nosec B324
        if prompt_key in gemini_prompt_cache:  # nosec B324
            logger.info(f"[{run_id} {request_id}] ✅ Cache hit for chunk {cid}")
            return gemini_prompt_cache[prompt_key]
        try:
            async with httpx.AsyncClient(http2=True, timeout=90) as client:
                response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                logger.error(
                    f"[{run_id} {request_id}] Gemini chunk {cid} failed: {response.status_code} - {response.text}"
                )
                return {"segments": []}

            result = response.json()
            candidate = result["candidates"][0]
            summary_text = candidate["content"]["parts"][0]["text"].strip()

            # Clean and parse JSON
            cleaned_summary = summary_text.strip()
            if cleaned_summary.startswith("```json"):
                cleaned_summary = cleaned_summary[7:]
            elif cleaned_summary.startswith("```"):
                cleaned_summary = cleaned_summary[3:]
            if cleaned_summary.endswith("```"):
                cleaned_summary = cleaned_summary[:-3]
            cleaned_summary = cleaned_summary.strip()

            summary_json = json.loads(cleaned_summary)

            gemini_prompt_cache[prompt_key] = summary_json
            return summary_json
        except Exception as e:
            logger.error(f"[{run_id} {request_id}] Gemini chunk {cid} parse error: {e}")
            return {"segments": []}

    # Run all chunks in parallel
    tasks = [process_chunk(chunk, cid) for cid, chunk in chunks]
    results = await asyncio.gather(*tasks)

    # Merge results, deduplicate by speaker+text
    merged_segments = []
    seen_texts = set()
    for result in results:
        for seg in result.get("segments", []):
            key = (seg["speaker"], seg["text"])
            if key not in seen_texts:
                seen_texts.add(key)
                merged_segments.append(seg)

    current_time = 0
    for seg in merged_segments:
        seg["start_timestamp"] = seconds_to_mmss(current_time)
        current_time += DEFAULT_SEGMENT_DURATION_GEMINI
        seg["end_timestamp"] = seconds_to_mmss(current_time)

    return {"segments": merged_segments}


# OPENAI SEGMENTATION
openai_prompt_cache = TTLCache(maxsize=5000, ttl=6 * 3600)


# @async_capture_span("segregate_with_openai")
async def segregate_with_openai(
    full_transcript: str, run_id: str, request_id: str
) -> Dict:
    """Segment transcript into categories using OpenAI API with parallel chunking"""
    if not OPENAI_API_KEY:
        return {"error": "OpenAI API key not found in environment variables"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    system_prompt = """
                    You are a helpful assistant that segments lawyer-client call transcripts. Your job is to divide the transcript into continuous, chronological segments, each belonging to one of four categories: Introduction_Greeting, Client_Problem_Questions, Legal_Advice_Options, and Next_Steps_Closing. Present all output in a structured JSON format.

                    Focus on conversation flow and speaker roles:
                    - Assign the speaker based on who is actually speaking ("lawyer" or "client"), not keywords alone.
                    - Maintain strict chronological order; never reorder segments.
                    - Segment categories should reflect content and context.
                    - Only assign "Introduction_Greeting" to opening greetings or small talk.
                    - Acknowledgments or fillers ("okay", "alright") should be categorized based on context.
                    - Ensure continuity across segments; consider previous segments for accurate flow.

                    Each segment should:
                    - Have one of the four segment_ids; do not create new ones.
                    - Be continuous and non-overlapping.
                    - Use MM:SS timestamps.
                    - Identify speaker as "lawyer" or "client".
                    - Return ONLY valid JSON; no explanations or markdown.
                    - Include empty segments if needed.
                    - Be separate per speaker snippet; do not combine multiple snippets.
                    - Follow the transcript order; do not merge snippets or reorder segments.
                    - Use best-guess judgment if a snippet is ambiguous.
"""

    # Split into chunks with overlap
    chunks = []
    i, chunk_id = 0, 1
    while i < len(full_transcript):
        chunk = full_transcript[i : i + CHUNK_SIZE_OPENAI]
        chunks.append((chunk_id, chunk))
        i += CHUNK_SIZE_OPENAI - OVERLAP_OPENAI
        chunk_id += 1

    logger.info(f"[{run_id} {request_id}] Transcript split into {len(chunks)} chunks.")

    # async worker for each chunk
    async def process_chunk(chunk_text: str, cid: int):
        user_prompt = f"""
        **IMPORTANT** : Use best-guess judgment for ambiguous snippets, but never swap lawyer/client roles till the call ends.

        Segment the following transcript chunk into JSON according to the rules.

        Each segment must include:
        - segment_id (one of the four categories above)
        - start_timestamp (MM:SS)
        - end_timestamp (MM:SS)
        - speaker ("lawyer" or "client", assigned **based on context**)
        - text (verbatim from that snippet)

        Segment categories:
        1. Introduction_Greeting
        2. Client_Problem_Questions
        3. Legal_Advice_Options
        4. Next_Steps_Closing

        Segment details:

        1. Introduction_Greeting
        - Client greetings, introduction
        - Lawyer greetings, introductions

        2. Client_Problem_Questions
        - Client statements describing problem, context, dates, documents
        - Initial context questions to understand the client’s problem
        - Lawyer clarifying questions
        - Exclude advice or solutions

        3. Legal_Advice_Options
        - Lawyer-provided advice, recommendations, or legal options
        - Include pros/cons, analysis of different approaches
        - Clients should not provide legal advice or solutions
        - Exclude greetings, client problem description, or closing statements

        4. Next_Steps_Closing
        - Follow-up instructions, deadlines, documents required, confirmations, closing remarks
        - Any final questions from client or lawyer

        Additional rules:
        - Do NOT merge multiple speaker snippets; each snippet should be a separate segment.
        - Preserve original order; do not reorder segments.
        - Ensure timestamps are continuous and non-overlapping.
        - Return ONLY properly formatted JSON.


        Speaker Assignment Rules (strict):
        1. Lawyer: Any segment that provides guidance, legal advice, recommendations, explanations, or options.
        2. Client: Any segment that describes problems, asks questions, expresses confusion, or reacts to lawyer guidance.
        3. Do not rely on words like "drive", "okay", or "yes" alone to assign speaker. Assign based on **who is contributing advice vs. seeking information**.
        4. Consider previous and following segments to maintain correct speaker roles and logical flow.
        5. Segments must **never swap roles mid-flow**. The client cannot give legal advice; the lawyer cannot describe their problem.
        6. Maintain chronological order.

        **IMPORTANT** - Ensure the entire transcript is segmented from start to finish, creating empty segments if a portion contains no meaningful content.

        Format strictly as:

        {{
            "segments": [
                {{
                    "segment_id": "",
                    "start_timestamp": "",
                    "end_timestamp": "",
                    "speaker": "",
                    "text": ""
                }}
            ]
        }}

        Transcript chunk (part {cid}):
        {chunk_text}
        """

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        payload_string = json.dumps(data, sort_keys=True)
        prompt_key = hashlib.md5(
            payload_string.encode("utf-8")
        ).hexdigest()  # nosec B324

        if prompt_key in openai_prompt_cache:
            logger.info(f"[{run_id} {request_id}] 💾 OpenAI cache hit for chunk {cid}")
            return openai_prompt_cache[prompt_key]

        try:
            async with httpx.AsyncClient(http2=True, timeout=120) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    data=json.dumps(data),
                )
                resp.raise_for_status()
                api_result = resp.json()
                summary_text = api_result["choices"][0]["message"]["content"].strip()

                # Clean JSON
                cleaned = summary_text
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                elif cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                try:
                    parsed = json.loads(cleaned)
                    openai_prompt_cache[prompt_key] = parsed
                    return parsed
                except json.JSONDecodeError as e:
                    logger.error(
                        f"[{run_id} {request_id}] JSON decode error in chunk {cid}: {e}"
                    )
                    return {"segments": []}
        except Exception as e:
            logger.error(
                f"[{run_id} {request_id}] OpenAI chunk {cid} request failed: {e}"
            )
            return {"segments": []}

    # Run in parallel
    results = await asyncio.gather(
        *[process_chunk(chunk, cid) for cid, chunk in chunks]
    )

    # Merge results + assign continuous timestamps
    merged_segments = []
    current_time = 0
    for result in results:
        for seg in result.get("segments", []):
            # Always keep every segment, don't deduplicate blindly
            seg["start_timestamp"] = seconds_to_mmss(current_time)
            current_time += DEFAULT_SEGMENT_DURATION_OPENAI
            seg["end_timestamp"] = seconds_to_mmss(current_time)
            merged_segments.append(seg)

    return {"segments": merged_segments}


# GEMINI EVALUATION
gemini_eval_cache = TTLCache(maxsize=2000, ttl=6 * 3600)


# @async_capture_span("evaluate_segments_with_gemini")
async def evaluate_segments_with_gemini(
    batch, batch_idx, run_id: str, request_id: str
) -> Dict:
    """
    Evaluate each segment using Gemini API according to the legal call quality rubric.
    Inputs:
        - segmented_json: output from `segregate_with_gemini` function
        - run_id, request_id: logging/tracking IDs
    Returns:
        - JSON with evaluations for each segment
    """
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not found in environment variables"}

    transcript_text = json.dumps(batch)
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"""
                            You are an expert legal consultation analyst. Your task is to evaluate each segment of a lawyer-client call transcript
                            according to the following rubric dimensions: Clarity, Tone & Empathy, Depth & Completeness.

                            ### Rubric:

                            Dimension | Score 1 – Very Poor | Score 2 – Poor | Score 3 – Average | Score 4 – Good | Score 5 – Excellent
                            --- | --- | --- | --- | --- | ---
                            **Clarity** | Uses mostly legal jargon; client appears lost. Lawyer misses greeting, problem summary, disclaimers, and next steps. Talk-time heavily imbalanced; no next steps or action communicated. | Explains partially, still unclear; some mandatory steps skipped; frequent interruptions or filler words; next steps partially communicated. | Explains in plain terms but skips details; most mandatory steps done; talk-time somewhat balanced; next steps mentioned but not fully clear. | Explains clearly using simple/bilingual-friendly terms, analogies/examples; all mandatory steps completed; balanced talk-time; next steps fully communicated. | Explains step-by-step in plain/simple terms; checks client understanding; all mandatory steps completed; next steps, proposal, required documents, and ETA clearly stated.
                            **Tone & Empathy** | Dismissive, rushed; interrupts client; authoritative or impatient; ignores emotional cues; no reassurance given. | Answers queries but without warmth; minimal acknowledgment of client’s stress or emotions; hierarchical tone. | Polite and professional; listens but limited empathy; hedging used occasionally; client feels “spoken to” rather than “spoken with.” | Respectful, empathetic; acknowledges client’s stress; active listening evident; hedges appropriately to avoid overconfidence. | Highly empathetic and patient; culturally sensitive; reassures client; actively listens; provides encouragement and confidence throughout call.
                            **Depth & Completeness** | Gives vague advice; fails to ask relevant questions; skips mandatory checklist items; misinforms or ignores potential risks. | Provides partial advice; misses several key questions; some checklist items ignored; minor inaccurate information or compliance issues. | Covers basics; most must-ask questions addressed; minor gaps in checklist; compliance mostly followed; limited risk warnings. | Detailed advice; asks all relevant questions; checklist fully completed; clearly mentions legal options, pros/cons, and risks; no misleading statements. | Comprehensive advice; all probing questions asked; checklist complete; explores alternative solutions (Lok Adalat, civil suit, police complaint); provides realistic expectations and clear next steps; fully compliant; no guarantees or misleading claims.

                            ### Evaluation Guidelines:

                            - Note: Each segment is an excerpt from the full call transcript, but should be evaluated as part of the conversation flow, not as an isolated or out-of-context statement.
                            - Evaluate each segment **in the context of previous segments**, so scoring considers continuity and consistency of the conversation.
                            - Evaluate segments **fairly**, avoiding overly harsh scoring for informal but understandable conversation.
                            - For lawyer segments, focus on whether advice is actionable, understandable, and consistent with prior guidance.
                            - For client segments, consider whether the questions or statements are clear **relative to earlier context**, not just isolated grammar or clarity.
                            - Treat Score 3 as **average communication** in context; 4–5 for consistently clear and empathetic responses; 1–2 for genuinely poor communication.
                            - Include constructive suggestions that help improve clarity, tone, or depth, considering prior conversation flow.


                            ### Instructions:

                            1. For each segment of the call, provide:
                            - segment_id
                            - speaker
                            - text
                            - clarity_score (1–5) and a **short chain-of-thought explanation** for why this score was assigned
                            - tone_score (1–5) and a **short chain-of-thought explanation**
                            - depth_score (1–5) and a **short chain-of-thought explanation**
                            - improvement_suggestions (optional) - constructive suggestions to improve clarity, tone, or depth

                            3. **Only include lawyer segments in the output. Ignore and skip all client segments.**
                            4. Evaluate each lawyer segment independently. For each score explanation, **use reasoning step-by-step** to justify the rating.
                            5. Return ONLY valid JSON. Do not include explanations outside the JSON.

                            Format the response exactly as:

                            {{
                                "evaluations": [
                                    {{
                                        "segment_id": "",
                                        "speaker": "",
                                        "text": "",
                                        "clarity_score": 0,
                                        "clarity_explanation": "",
                                        "tone_score": 0,
                                        "tone_explanation": "",
                                        "depth_score": 0,
                                        "depth_explanation": ""
                                    }}
                                ]
                            }}

                            Here are the segments:
                            {transcript_text}
                        """
                    }
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    url = f"{GEMINI_GENERATE_URL}?key={GEMINI_API_KEY}"

    prompt_str = payload["contents"][0]["parts"][0]["text"]
    prompt_key = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()  # nosec B324

    if prompt_key in gemini_eval_cache:
        logger.info(
            f"[{run_id} {request_id}] 💾 Cache hit for Gemini evaluation batch {batch_idx}"
        )
        return gemini_eval_cache[prompt_key]

    try:
        async with httpx.AsyncClient(http2=True, timeout=120) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            logger.error(
                f"[{run_id} {request_id}] Gemini API request failed: {response.status_code} - {response.text}"
            )
            return {
                "error": f"Gemini API request failed: {response.status_code} - {response.text}"
            }

        result = response.json()
        candidate = result["candidates"][0]
        eval_text = candidate["content"]["parts"][0]["text"].strip()

        # Clean and parse JSON
        cleaned_eval = eval_text.strip()
        if cleaned_eval.startswith("```json"):
            cleaned_eval = cleaned_eval[7:]
        elif cleaned_eval.startswith("```"):
            cleaned_eval = cleaned_eval[3:]
        if cleaned_eval.endswith("```"):
            cleaned_eval = cleaned_eval[:-3]
        cleaned_eval = cleaned_eval.strip()

        eval_json = json.loads(cleaned_eval)

        gemini_eval_cache[prompt_key] = eval_json
        return eval_json
    except Exception as e:
        logger.error(
            f"[{run_id} {request_id}] Gemini evaluation request failed for batch {batch_idx}: {e}"
        )
        return {"evaluations": []}


# OPENAI EVALUATION
openai_eval_cache = TTLCache(maxsize=2000, ttl=6 * 3600)


# @async_capture_span("evaluate_segments_with_openai")
async def evaluate_segments_with_openai(
    batch, batch_idx, run_id: str, request_id: str
) -> Dict:
    """
    Evaluate each segment using OpenAI API according to the legal call quality rubric.
    Inputs:
        - segmented_json: output from `segregate_with_openai` function
        - run_id, request_id: logging/tracking IDs
    Returns:
        - JSON with evaluations for each lawyer segment
    """
    if not OPENAI_API_KEY:
        return {"error": "OpenAI API key not found in environment variables"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    transcript_text = json.dumps(batch)

    system_prompt = """You are an expert legal consultation analyst.
    Your task is to evaluate each segment of a lawyer-client call transcript
    according to the following rubric dimensions: Clarity, Tone & Empathy, Depth & Completeness.

    **IMPORTANT**: Don't mix client segments with lawyer segments.
    **Only evaluate lawyer segments. Skip and ignore all client segments.**

    ### Rubric:

    Dimension | Score 1 – Very Poor | Score 2 – Poor | Score 3 – Average | Score 4 – Good | Score 5 – Excellent
    --- | --- | --- | --- | --- | ---
    **Clarity** | Uses mostly legal jargon; client appears lost. Lawyer misses greeting, problem summary, disclaimers, and next steps. Talk-time heavily imbalanced; no next steps or action communicated. | Explains partially, still unclear; some mandatory steps skipped; frequent interruptions or filler words; next steps partially communicated. | Explains in plain terms but skips details; most mandatory steps done; talk-time somewhat balanced; next steps mentioned but not fully clear. | Explains clearly using simple/bilingual-friendly terms, analogies/examples; all mandatory steps completed; balanced talk-time; next steps fully communicated. | Explains step-by-step in plain/simple terms; checks client understanding; all mandatory steps completed; next steps, proposal, required documents, and ETA clearly stated.
    **Tone & Empathy** | Dismissive, rushed; interrupts client; authoritative or impatient; ignores emotional cues; no reassurance given. | Answers queries but without warmth; minimal acknowledgment of client’s stress or emotions; hierarchical tone. | Polite and professional; listens but limited empathy; hedging used occasionally; client feels “spoken to” rather than “spoken with.” | Respectful, empathetic; acknowledges client’s stress; active listening evident; hedges appropriately to avoid overconfidence. | Highly empathetic and patient; culturally sensitive; reassures client; actively listens; provides encouragement and confidence throughout call.
    **Depth & Completeness** | Gives vague advice; fails to ask relevant questions; skips mandatory checklist items; misinforms or ignores potential risks. | Provides partial advice; misses several key questions; some checklist items ignored; minor inaccurate information or compliance issues. | Covers basics; most must-ask questions addressed; minor gaps in checklist; compliance mostly followed; limited risk warnings. | Detailed advice; asks all relevant questions; checklist fully completed; clearly mentions legal options, pros/cons, and risks; no misleading statements. | Comprehensive advice; all probing questions asked; checklist complete; explores alternative solutions (Lok Adalat, civil suit, police complaint); provides realistic expectations and clear next steps; fully compliant; no guarantees or misleading claims.

    ### Evaluation Guidelines:

    - Note: Each segment is an excerpt from the full call transcript, but should be evaluated as part of the conversation flow, not as an isolated or out-of-context statement.
    - Evaluate each segment **in the context of previous segments**, so scoring considers continuity and consistency of the conversation.
    - Evaluate segments **fairly**, avoiding overly harsh scoring for informal but understandable conversation.
    - For lawyer segments, focus on whether advice is actionable, understandable, and consistent with prior guidance.
    - For client segments, consider whether the questions or statements are clear **relative to earlier context**, not just isolated grammar or clarity.
    - Treat Score 3 as **average communication** in context; 4–5 for consistently clear and empathetic responses; 1–2 for genuinely poor communication.
    - Include constructive suggestions that help improve clarity, tone, or depth, considering prior conversation flow.

    ### Instructions:

    1. For each segment of the call, provide:
    - segment_id
    - speaker
    - text
    - clarity_score (1–5) and a **short chain-of-thought explanation** based on context
    - tone_score (1–5) and a **short chain-of-thought explanation** based on context
    - depth_score (1–5) and a **short chain-of-thought explanation** based on context
    - improvement_suggestions (optional) - constructive suggestions to improve clarity, tone, or depth

    2. **Only include lawyer segments in the output. Ignore and skip all client segments.**
    3. **Evaluate each segment independently but consider all previous segments for context.**
    4. **Return ONLY valid JSON. Do not include explanations outside JSON.** """

    user_prompt = f"""
    Evaluate these transcript segments strictly by the rules above:

    {transcript_text}

    Format strictly as:
    {{
        "evaluations": [
            {{
                "segment_id": "",
                "speaker": "",
                "text": "",
                "clarity_score": 0,
                "clarity_explanation": "",
                "tone_score": 0,
                "tone_explanation": "",
                "depth_score": 0,
                "depth_explanation": ""
            }}
        ]
    }}
    """

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    payload_string = json.dumps(data, sort_keys=True)
    prompt_key = hashlib.md5(payload_string.encode("utf-8")).hexdigest()  # nosec B324

    if prompt_key in openai_eval_cache:
        logger.info(
            f"[{run_id} {request_id}] 💾 Cache hit for OpenAI evaluation batch {batch_idx}"
        )
        return openai_eval_cache[prompt_key]

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(data),
            )
            resp.raise_for_status()
            api_result = resp.json()
            eval_text = api_result["choices"][0]["message"]["content"].strip()

            # Clean JSON
            cleaned_eval = eval_text
            if cleaned_eval.startswith("```json"):
                cleaned_eval = cleaned_eval[7:]
            elif cleaned_eval.startswith("```"):
                cleaned_eval = cleaned_eval[3:]
            if cleaned_eval.endswith("```"):
                cleaned_eval = cleaned_eval[:-3]
            cleaned_eval = cleaned_eval.strip()

            eval_json = json.loads(cleaned_eval)
            return eval_json

    except Exception as e:
        logger.error(
            f"[{run_id} {request_id}] ⚠️ OpenAI batch {batch_idx} failed: {e}. Retrying once..."
        )

        # 🔁 Retry logic — one more attempt
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    data=json.dumps(data),
                )
                resp.raise_for_status()
                api_result = resp.json()
                eval_text = api_result["choices"][0]["message"]["content"].strip()
                cleaned_eval = eval_text.strip("```json").strip("```").strip()
                eval_json = json.loads(cleaned_eval)

                openai_eval_cache[prompt_key] = eval_json
                return eval_json
        except Exception as e2:
            logger.error(
                f"[{run_id} {request_id}] ❌ OpenAI retry also failed for batch {batch_idx}: {e2}"
            )
            return {"evaluations": []}


# Wrapper to process all batches with fallback
async def evaluate_all_batches(segments, run_id: str, request_id: str):
    max_batch_size = 20
    batches = [
        segments[i : i + max_batch_size]
        for i in range(0, len(segments), max_batch_size)
    ]

    async def process_batch(batch, batch_idx, provider):
        logger.info(
            f"[{run_id} {request_id}] Processing batch {batch_idx+1}/{len(batches)} with {provider}"
        )

        try:
            if provider == "gemini":
                result = await evaluate_segments_with_gemini(
                    batch, batch_idx, run_id, request_id
                )
            else:
                result = await evaluate_segments_with_openai(
                    batch, batch_idx, run_id, request_id
                )

            evaluations = result.get("evaluations", [])

            # 🔄 fallback if no results
            if not evaluations:
                fallback = "openai" if provider == "gemini" else "gemini"
                logger.info(
                    f"[{run_id} {request_id}] {provider} failed for batch {batch_idx}, falling back to {fallback}..."
                )
                if fallback == "gemini":
                    result = await evaluate_segments_with_gemini(
                        batch, batch_idx, run_id, request_id
                    )
                else:
                    result = await evaluate_segments_with_openai(
                        batch, batch_idx, run_id, request_id
                    )
                evaluations = result.get("evaluations", [])

            return evaluations

        except Exception as e:
            logger.error(
                f"[{run_id} {request_id}] {provider} crashed for batch {batch_idx}: {e}"
            )
            return []

    # Example: alternate batches between Gemini and OpenAI
    tasks = [
        process_batch(batch, idx, "gemini" if idx % 2 == 0 else "openai")
        for idx, batch in enumerate(batches)
    ]

    results = await asyncio.gather(*tasks)

    # flatten results
    all_results = [ev for batch_result in results for ev in batch_result]

    return {"evaluations": all_results}


"""Quering the texts and getting similar texts from top calls"""

# PINECONE SIMILARITY SEARCH
embedding_cache = TTLCache(maxsize=10000, ttl=6 * 3600)  # 6 hr cache
pinecone_cache = TTLCache(maxsize=10000, ttl=6 * 3600)


# @async_capture_span("get_similar_segments_from_pinecone")
async def get_similar_segments_from_pinecone(
    incoming_call_json: Dict[str, Any], pinecone_index, top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Batch process segments:
    - Compute embeddings in batches
    - Query Pinecone in batches
    - Return structured results
    """
    segments = incoming_call_json.get("evaluations", [])
    logger.info(f"Processing {len(segments)} segments from incoming call")

    # --- Precompute embeddings in batches ---
    texts = [seg["text"] for seg in segments]
    emb_keys = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts]  # nosec B324

    embeddings = []
    missing_indices = []
    for i, emb_key in enumerate(emb_keys):
        if emb_key in embedding_cache:
            embeddings.append(embedding_cache[emb_key])
            logger.info(f"✅ Embedding cache hit for seg {segments[i]['segment_id']}")
        else:
            embeddings.append(None)
            missing_indices.append(i)

    # Batch embedding API call for missing ones
    if missing_indices:
        batch_texts = [texts[i] for i in missing_indices]
        batch_embeddings = await get_embedding_batch(batch_texts)
        for idx, emb in zip(missing_indices, batch_embeddings):
            embeddings[idx] = emb
            embedding_cache[emb_keys[idx]] = emb

    # --- Prepare Pinecone queries ---
    results = []
    sem = asyncio.Semaphore(20)  # limit concurrency

    async def query_segment(i):
        seg = segments[i]
        segment_id = seg["segment_id"]
        call_text = seg["text"]

        try:
            async with sem:
                query_key = f"{emb_keys[i]}_{segment_id}_{top_k}"
                if query_key in pinecone_cache:
                    query_response = pinecone_cache[query_key]
                    logger.info(f"✅ Pinecone cache hit for seg {segment_id}")
                else:
                    # Use thread executor for sync query
                    loop = asyncio.get_event_loop()
                    query_response = await loop.run_in_executor(
                        None,
                        lambda: pinecone_index.query(
                            vector=embeddings[i],
                            top_k=top_k,
                            include_metadata=True,
                            include_values=False,
                            filter={"segment_id": {"$eq": segment_id}},
                        ),
                    )
                    pinecone_cache[query_key] = query_response

            reference_data = [
                {
                    "text": m["metadata"]["text"],
                    "clarity_score": m["metadata"].get("clarity_score"),
                    "tone_score": m["metadata"].get("tone_score"),
                    "depth_score": m["metadata"].get("depth_score"),
                    "segment_id": m["metadata"].get("segment_id"),
                }
                for m in query_response.get("matches", [])
                if "metadata" in m and "text" in m["metadata"]
            ]

            return {
                "call": {
                    "segment_id": segment_id,
                    "texts": [call_text],
                    "clarity_score": seg.get("clarity_score"),
                    "tone_score": seg.get("tone_score"),
                    "depth_score": seg.get("depth_score"),
                    "clarity_explanation": seg.get("clarity_explanation"),
                    "tone_explanation": seg.get("tone_explanation"),
                    "depth_explanation": seg.get("depth_explanation"),
                },
                "reference": {"texts": reference_data},
            }

        except Exception as e:
            logger.error(f"⚠️ Skipping segment {segment_id}: {e}")
            return None

    # Run all segments concurrently
    tasks = [query_segment(i) for i in range(len(segments))]
    results = await asyncio.gather(*tasks)

    return [r for r in results if r]


def build_transcript_with_segments(
    segments: List[dict], is_lawyer_call: bool = False
) -> str:
    """
    Builds a formatted transcript string from a list of segment dictionaries.

    Args:
        segments: A list of dictionaries, where each dictionary represents a segment.
        is_lawyer_call: A boolean flag to indicate if the segments are from the lawyer's call.
                        This determines whether to include explanations in the output.

    Returns:
        A string representing the formatted transcript.
    """
    parts = []

    # Process segments based on whether they are from the lawyer's call
    if is_lawyer_call:
        for seg in segments:
            seg_id = seg.get("segment_id", "unknown")
            texts = seg.get("texts", [])
            if not texts:
                continue

            header_parts = [f"### Segment ID: {seg_id}"]
            clarity_score = seg.get("clarity_score", None)
            tone_score = seg.get("tone_score", None)
            depth_score = seg.get("depth_score", None)

            if clarity_score is not None:
                header_parts.append(f"Clarity Score: {clarity_score}/5")
            if tone_score is not None:
                header_parts.append(f"Tone Score: {tone_score}/5")
            if depth_score is not None:
                header_parts.append(f"Depth Score: {depth_score}/5")

            parts.append(" - ".join(header_parts))

            clarity_explanation = seg.get("clarity_explanation", "")
            tone_explanation = seg.get("tone_explanation", "")
            depth_explanation = seg.get("depth_explanation", "")
            if clarity_explanation:
                parts.append(f"Clarity Explanation: {clarity_explanation}")
            if tone_explanation:
                parts.append(f"Tone Explanation: {tone_explanation}")
            if depth_explanation:
                parts.append(f"Depth Explanation: {depth_explanation}")

            for t in texts:
                if t.strip():
                    parts.append(t)
    else:
        # This branch handles the reference segments
        for seg in segments:
            texts = seg.get("texts", [])
            if isinstance(texts, list) and all(isinstance(t, dict) for t in texts):
                for ref_data in texts:
                    ref_text = ref_data.get("text", "")
                    if ref_text and isinstance(ref_text, str) and ref_text.strip():
                        header_parts = [
                            f"### Reference ID: {ref_data.get('segment_id', 'unknown')}"
                        ]
                        clarity_score = ref_data.get("clarity_score", None)
                        tone_score = ref_data.get("tone_score", None)
                        depth_score = ref_data.get("depth_score", None)

                        if clarity_score is not None:
                            header_parts.append(f"Clarity Score: {clarity_score}/5")
                        if tone_score is not None:
                            header_parts.append(f"Tone Score: {tone_score}/5")
                        if depth_score is not None:
                            header_parts.append(f"Depth Score: {depth_score}/5")

                        parts.append(" - ".join(header_parts))
                        parts.append(ref_text)
            else:
                # Fallback for unexpected format
                # Assuming this won't be hit with the current data
                for t in texts:
                    if isinstance(t, str) and t.strip():
                        parts.append(t)

    return "\n".join(parts)


# OPENAI SUGGESTIONS
openai_improvement_cache = TTLCache(maxsize=500, ttl=12 * 3600)
"""Using the texts and similar texts to get improvements to be made in the call with help of prompt engineering"""


# @async_capture_span("suggest_lawyer_improvements_with_openai")
async def suggest_lawyer_improvements_with_openai(
    segmented_json: List[Dict], run_id: str, request_id: str
) -> Dict:
    """
    Suggest improvements for the overall lawyer call by comparing the full call transcript
    (all segments combined) with reference transcripts.

    Inputs:
        - segmented_json: list of dicts with {call: [...], reference: [...]}
        - run_id, request_id: logging/tracking IDs
    Returns:
        - Dict with improvement summary text (single flowing summary, no bullet points)
    """

    if not OPENAI_API_KEY:
        return {"error": "OpenAI API key not found in environment variables"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    # --- 🔹 Combine all lawyer & reference segments into full transcripts ---

    all_lawyer_segments = []
    all_reference_segments = []
    for pair in segmented_json:
        call_seg = pair.get("call")
        ref_seg = pair.get("reference")

        if call_seg:
            all_lawyer_segments.append(call_seg)  # not extend()
        if ref_seg:
            all_reference_segments.append(ref_seg)  # not extend()
    call_text = build_transcript_with_segments(all_lawyer_segments, is_lawyer_call=True)
    reference_text = build_transcript_with_segments(
        all_reference_segments, is_lawyer_call=False
    )

    # --- Prompts ---
    system_prompt = """You are a professional call evaluator and coach for lawyers in India. Your primary goal is to provide a comprehensive, structured coaching summary based on a given lawyer-client conversation.

You will respond with a single, valid JSON object that contains the following keys:
- "strengths": What the lawyer did well.
- "clarity": Suggestions to improve clarity.
- "tone": Suggestions to improve tone.
- "depth": Suggestions to improve the depth of legal advice.

Guidelines for your response:
- Address the lawyer as "you".
- Focus on areas with low scores if provided in the transcript.
- Use the reference transcripts as examples of strong communication to support your suggestions.
- When suggesting improvements, quote short snippets from the lawyer's transcript and show possible rephrasing.
- Respect Indian conversational norms and professional call etiquette.
- Do not invent legal facts or discuss legal outcomes; critique only communication style.
- Each key's content must be a single, flowing summary, not a list."""

    user_prompt = f"""
Here is the lawyer's call transcript with detailed segment analysis:

{call_text}

Reference transcripts for comparison:

{reference_text}
"""

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }

    payload_string = json.dumps(data, sort_keys=True)
    prompt_key = hashlib.md5(payload_string.encode("utf-8")).hexdigest()  # nosec B324

    if prompt_key in openai_improvement_cache:
        logger.info(
            f"[{run_id} {request_id}] 💾 Cache hit for lawyer improvement suggestion"
        )
        return openai_improvement_cache[prompt_key]

    def normalize_quotes(text: str) -> str:
        replacements = {"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"'}
        return re.sub(
            "|".join(map(re.escape, replacements.keys())),
            lambda m: replacements[m.group(0)],
            text,
        )

    try:
        async with httpx.AsyncClient(http2=True, timeout=120) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                data=json.dumps(data),
            )
            resp.raise_for_status()
            api_result = resp.json()

            # --- Extract raw text ---
            suggestion_text = api_result["choices"][0]["message"]["content"].strip()

            # --- Clean output ---
            cleaned = suggestion_text
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            cleaned = normalize_quotes(cleaned).strip()

            # --- Try JSON parse for the new structured format ---
            try:
                parsed = json.loads(cleaned)
                expected_keys = ["strengths", "clarity", "tone", "depth"]

                # Validate that the parsed object is a dictionary with all expected keys
                if isinstance(parsed, dict) and all(
                    key in parsed for key in expected_keys
                ):
                    return {"suggestions": parsed}
                else:
                    # Fallback if parsing works but the structure is wrong
                    logger.warning(
                        f"[{run_id} {request_id}] OpenAI returned an unexpected JSON structure."
                    )
                    return {"suggestions": {key: "" for key in expected_keys}}
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails entirely
                logger.warning(
                    f"[{run_id} {request_id}] OpenAI returned unparseable JSON."
                )
                return {"suggestions": {key: "" for key in expected_keys}}
        openai_improvement_cache[prompt_key] = parsed
    except Exception as e:
        logger.error(
            f"[{run_id} {request_id}] OpenAI improvement suggestion request failed: {e}"
        )
        # Return a default empty structure on any API or network error
        return {
            "suggestions": {"strengths": "", "clarity": "", "tone": "", "depth": ""}
        }


# GEMINI SUGGESTIONS
gemini_improvement_cache = TTLCache(maxsize=500, ttl=12 * 3600)


# @async_capture_span("suggest_lawyer_improvements_with_gemini")
async def suggest_lawyer_improvements_with_gemini(
    segmented_json: List[Dict], run_id: str, request_id: str
) -> Dict:
    """
    Suggest improvements for lawyer communication in a call by comparing with reference transcripts.
    Inputs:
        - segmented_json: the incoming call transcript (segmented, includes lawyer + client)
        - reference_json: reference transcripts (segmented, includes lawyer + client)
        - run_id, request_id: logging/tracking IDs
    Returns:
        - JSON: { "suggestions": "..." }  (a single summary text)
    """

    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not found in environment variables"}

    GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    all_lawyer_segments = []
    all_reference_segments = []
    for pair in segmented_json:
        call_seg = pair.get("call")
        ref_seg = pair.get("reference")

        if call_seg:
            all_lawyer_segments.append(call_seg)
        if ref_seg:
            # Note: ref_seg is a dict. The 'texts' key inside it is a list of dicts.
            all_reference_segments.append(ref_seg)

    call_text = build_transcript_with_segments(all_lawyer_segments, is_lawyer_call=True)
    reference_text = build_transcript_with_segments(
        all_reference_segments, is_lawyer_call=False
    )

    # --- Prompt ---
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"""
                    You are a professional call evaluator and coach for lawyers in India.
                    Your role is to give clear, actionable feedback to the lawyer on how you can improve your communication skills, using the provided call analysis and strong reference transcripts.

                    ### Guidelines:
                    - Address the lawyer as "you".
                    - Structure your response as JSON with keys: "strengths", "clarity", "tone", "depth".
                    - For each key, provide a single, flowing summary of suggestions related to that specific area.
                    - For each key, provide a single flowing summary highlighting strengths or suggestions.
                    - Use the segment scores and reference transcripts to inform feedback. Quote short snippets and suggest clear rephrasings.
                    - Focus only on communication style; do not invent legal facts or outcomes.
                    - Provide concise, clear, actionable suggestions. Include at least one concrete rephrased example per feedback category where applicable.

                    ### Lawyer's Call Transcript with Analysis:
                    {call_text}

                    ### Strong Reference Transcripts:
                    {reference_text}

                    ### Response format:
                    {{
                    "suggestions": {{
                        "strengths": "<single flowing summary highlighting what was done well>",
                        "clarity": "<single flowing summary of clarity suggestions>",
                        "tone": "<single flowing summary of tone suggestions>",
                        "depth": "<single flowing summary of depth suggestions>"
                    }}
                    }}
                    """
                    }
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    url = f"{GEMINI_GENERATE_URL}?key={GEMINI_API_KEY}"

    prompt_str = payload["contents"][0]["parts"][0]["text"]
    prompt_key = hashlib.md5(prompt_str.encode("utf-8")).hexdigest()  # nosec B324

    if prompt_key in gemini_improvement_cache:
        logger.info(
            f"[{run_id} {request_id}] 💾 Cache hit for lawyer improvement suggestion"
        )
        return gemini_improvement_cache[prompt_key]

    try:
        async with httpx.AsyncClient(http2=True, timeout=120) as client:
            response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            logger.error(
                f"[{run_id} {request_id}] Gemini API request failed: {response.status_code} - {response.text}"
            )
            return {
                "error": f"Gemini API request failed: {response.status_code} - {response.text}"
            }

        result = response.json()
        candidate = result["candidates"][0]
        eval_text = candidate["content"]["parts"][0]["text"].strip()

    except Exception as e:
        logger.error(f"[{run_id} {request_id}] Gemini improvement request failed: {e}")
        return {"suggestions": ""}

    def normalize_quotes(text: str) -> str:
        replacements = {"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"'}
        return re.sub(
            "|".join(map(re.escape, replacements.keys())),
            lambda m: replacements[m.group(0)],
            text,
        )

    # --- 🔹 Clean model output ---
    cleaned_eval = eval_text.strip()

    # Strip markdown fences
    if cleaned_eval.startswith("```json"):
        cleaned_eval = cleaned_eval[7:]
    elif cleaned_eval.startswith("```"):
        cleaned_eval = cleaned_eval[3:]
    if cleaned_eval.endswith("```"):
        cleaned_eval = cleaned_eval[:-3]

    cleaned_eval = normalize_quotes(cleaned_eval).strip()

    # Try JSON parse
    try:
        eval_json = json.loads(cleaned_eval)
        if not isinstance(eval_json, dict):
            # if it's not already a dict, wrap it
            eval_json = {"suggestions": str(eval_json)}
    except Exception:
        # fallback: return raw text as suggestions
        logger.warning(f"[{run_id} {request_id}] Failed to parse JSON from Gemini")
        eval_json = {"suggestions": cleaned_eval}

    gemini_improvement_cache[prompt_key] = eval_json

    return eval_json


async def select_provider(provider_preference="OPENAI"):
    """Decide which provider to use based on current load."""
    if provider_preference.upper() == "OPENAI":
        if openai_semaphore.locked():
            # OpenAI is busy → fallback to Gemini
            return "GEMINI"
        return "OPENAI"
    elif provider_preference.upper() == "GEMINI":
        if gemini_semaphore.locked():
            # Gemini is busy → fallback to OpenAI
            return "OPENAI"
        return "GEMINI"
    return provider_preference


async def async_suggestion_with_limit(transcript, run_id, request_id, provider):
    provider = await select_provider(provider)

    semaphore = openai_semaphore if provider == "OPENAI" else gemini_semaphore

    async with semaphore:
        result = await async_suggestion(
            transcript=transcript,
            run_id=run_id,
            request_id=request_id,
            provider=provider,
        )
    return result


# @async_capture_span("async_suggestion")
async def async_suggestion(
    transcript: str, run_id: str, request_id: str, provider: str
):
    full_transcript = transcript
    logger.info(
        f"[{run_id} {request_id}] Starting suggestion pipeline with provider: {provider}"
    )
    # --- Step 1: Segregation with fallback ---
    try:
        start_seg = time.time()
        logger.info(f"[{run_id} {request_id}] Segregating full transcript")
        if provider == "GEMINI":
            try:
                result = await segregate_with_gemini(
                    full_transcript, run_id, request_id
                )
                if not result.get("segments"):
                    raise ValueError("Empty segments from Gemini, using fallback.")
            except Exception as e:
                logger.error(
                    f"Segregation with Gemini failed: {e}, falling back to OpenAI."
                )
                result = await segregate_with_openai(
                    full_transcript, run_id, request_id
                )
        elif provider == "OPENAI":
            try:
                result = await segregate_with_openai(
                    full_transcript, run_id, request_id
                )
                if not result.get("segments"):
                    raise ValueError("Empty segments from OpenAI, using fallback.")
            except Exception as e:
                logger.error(
                    f"Segregation with OpenAI failed: {e}, falling back to Gemini."
                )
                result = await segregate_with_gemini(
                    full_transcript, run_id, request_id
                )
        else:
            return {"error": "Invalid provider specified"}

        end_seg = time.time()
        if result.get("segments"):
            logger.info(
                f"[{run_id} {request_id}] Segregation completed in {end_seg - start_seg:.2f} seconds"
            )
        else:
            logger.info(f"[{run_id} {request_id}] returned empty segments")

    except Exception as e:
        logger.error(f"[{run_id} {request_id}] Segregation failed: {e}")
        return {"error": f"Segregation failed: {str(e)}"}

    # --- Step 2: Evaluation with fallback ---
    try:
        start_eval = time.time()
        logger.info(f"[{run_id} {request_id}] Starting evaluation with batching")
        score_result = await evaluate_all_batches(
            result["segments"], run_id=run_id, request_id=request_id
        )
        end_eval = time.time()
        if not score_result.get("evaluations") or len(score_result["evaluations"]) == 0:
            logger.warning(f"[{run_id} {request_id}] Evaluation returned empty result.")
            raise ValueError("Evaluation returned empty result")
        else:
            logger.info(
                f"[{run_id} {request_id}] Evaluation completed in {end_eval - start_eval:.2f} seconds"
            )
    except Exception as e:
        logger.error(f"[{run_id} {request_id}] Evaluation failed: {e}")
        return {"error": f"Evaluation failed: {str(e)}"}

    # --- Step 3: Retrieve similar segments ---
    try:
        start_pinecone = time.time()
        logger.info(f"[{run_id} {request_id}] Starting Pinecone similarity retrieval")
        index_name = "call-transcripts-index"
        pinecone_index = pc.Index(index_name)
        res = await get_similar_segments_from_pinecone(score_result, pinecone_index)
        end_pinecone = time.time()
        if len(res) == 0:
            logger.warning(
                f"[{run_id} {request_id}] Pinecone similarity retrieval returned empty result."
            )
            raise ValueError("Pinecone similarity retrieval returned empty result")
        else:
            logger.info(
                f"[{run_id} {request_id}] Pinecone similarity retrieval completed in {end_pinecone - start_pinecone:.2f} seconds"
            )

    except Exception as e:
        logger.error(
            f"[{run_id} {request_id}] Pinecone similarity retrieval failed: {e}"
        )
        return {"error": f"Pinecone similarity retrieval failed: {str(e)}"}

    # --- Step 4: Suggestion with fallback ---
    try:
        start_suggest = time.time()
        logger.info(f"[{run_id} {request_id}] Starting suggestion generation")
        if provider == "GEMINI":
            try:
                suggestion = await suggest_lawyer_improvements_with_gemini(
                    res, run_id, request_id
                )
                if not suggestion.get("suggestions"):
                    raise ValueError("Empty suggestions from Gemini, using fallback.")
            except Exception as e:
                logger.error(
                    f"Suggestion with Gemini failed: {e}, falling back to OpenAI."
                )
                suggestion = await suggest_lawyer_improvements_with_openai(
                    res, run_id, request_id
                )
        else:
            try:
                suggestion = await suggest_lawyer_improvements_with_openai(
                    res, run_id, request_id
                )
                if not suggestion.get("suggestions"):
                    raise ValueError("Empty suggestions from OpenAI, using fallback.")
            except Exception as e:
                logger.error(
                    f"Suggestion with OpenAI failed: {e}, falling back to Gemini."
                )
                suggestion = await suggest_lawyer_improvements_with_gemini(
                    res, run_id, request_id
                )
        end_suggest = time.time()
        if not suggestion.get("suggestions"):
            logger.warning(
                f"[{run_id} {request_id}] Suggestion generation returned empty result."
            )
            raise ValueError("Suggestion generation returned empty result")
        else:
            logger.info(
                f"[{run_id} {request_id}] Suggestion generation completed in {end_suggest - start_suggest:.2f} seconds"
            )
        return suggestion

    except Exception as e:
        logger.error(f"[{run_id} {request_id}] Suggestion generation failed: {e}")
        return {"error": f"Suggestion generation failed: {str(e)}"}
