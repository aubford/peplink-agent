import httpx
import json
import os
import logging
from typing import Iterator, AsyncIterator, Union
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = os.path.join(LOG_DIR, "openai_trace.log")

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("openai_trace")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s | %(message)s")
file_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(file_handler)


def _is_streaming_request(request_body: str) -> bool:
    """Check if the request is asking for streaming response."""
    try:
        body_json = json.loads(request_body)
        return body_json.get("stream", False) is True
    except:
        return False


def _parse_streaming_content(content: str) -> str:
    """Parse all streaming content from an OpenAI response."""
    try:

        # OpenAI streaming format: data: {...}\n\ndata: {...}\n\ndata: [DONE]\n\n
        lines = content.strip().split('\n')
        collected_chunks = []

        for line in lines:
            if line.startswith('data: ') and not line.endswith('[DONE]'):
                try:
                    data_content = line[6:]  # Remove "data: " prefix
                    chunk_json = json.loads(data_content)

                    # Extract content from different types of chunks
                    if 'choices' in chunk_json and chunk_json['choices']:
                        choice = chunk_json['choices'][0]
                        if 'delta' in choice and 'content' in choice['delta']:
                            if choice['delta']['content']:
                                collected_chunks.append(choice['delta']['content'])
                        elif 'message' in choice and 'content' in choice['message']:
                            if choice['message']['content']:
                                collected_chunks.append(choice['message']['content'])
                except json.JSONDecodeError:
                    continue

        complete_content = ''.join(collected_chunks)

        if complete_content:
            # Return just the complete response content
            return complete_content
        else:
            # If we couldn't parse chunks, return indication of failure
            return "[Could not parse streaming response content]"

    except Exception as e:
        logger.warning(f"Failed to collect streaming content: {e}")
        return "[Failed to collect streaming content]"


class LoggingTransport(httpx.BaseTransport):
    """Sync transport for logging OpenAI requests and responses."""

    def __init__(self, wrapped: httpx.BaseTransport):
        self._wrapped = wrapped

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        # Log the request
        try:
            body = request.content.decode("utf-8")
            pretty_body = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
            logger.info(f"REQUEST to {request.url}:\n{pretty_body}")

            # Check if this is a streaming request
            is_streaming = _is_streaming_request(body)
        except Exception as e:
            logger.warning(f"Failed to log request body: {e}")
            is_streaming = False

        response = self._wrapped.handle_request(request)

        # Handle response logging
        try:
            if is_streaming:
                # For streaming requests, collect and log the complete response
                logger.info(
                    f"RESPONSE ({response.status_code}) [STREAMING]: Collecting stream for {request.url}"
                )

                # Read the content once
                original_content = response.read()

                # Parse the streaming content
                complete_content = _parse_streaming_content(
                    original_content.decode('utf-8')
                )
                logger.info(
                    f"RESPONSE ({response.status_code}) [COMPLETE STREAM]:\n{complete_content}"
                )

                # Return response with the content we read
                return httpx.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=original_content,
                    request=request,
                )
            else:
                # For non-streaming responses, log immediately
                content_bytes = response.read()
                content_str = content_bytes.decode("utf-8")

                try:
                    parsed_json = json.loads(content_str)
                    pretty_content = json.dumps(
                        parsed_json, indent=2, ensure_ascii=False
                    )
                    logger.info(f"RESPONSE ({response.status_code}):\n{pretty_content}")
                except json.JSONDecodeError:
                    logger.info(
                        f"RESPONSE ({response.status_code}) [NON-JSON]:\n{content_str}"
                    )

                return httpx.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=content_bytes,
                    request=request,
                )
        except Exception as e:
            logger.warning(f"Failed to log response body: {e}")
            return response


class AsyncLoggingTransport(httpx.AsyncBaseTransport):
    """Async transport for logging OpenAI requests and responses."""

    def __init__(self, wrapped: httpx.AsyncBaseTransport):
        self._wrapped = wrapped

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Log the request
        try:
            body = request.content.decode("utf-8")
            pretty_body = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
            logger.info(f"ASYNC REQUEST to {request.url}:\n{pretty_body}")

            # Check if this is a streaming request
            is_streaming = _is_streaming_request(body)
        except Exception as e:
            logger.warning(f"Failed to log async request body: {e}")
            is_streaming = False

        response = await self._wrapped.handle_async_request(request)

        # Handle response logging
        try:
            if is_streaming:
                # For streaming requests, collect and log the complete response
                logger.info(
                    f"ASYNC RESPONSE ({response.status_code}) [STREAMING]: Collecting stream for {request.url}"
                )

                # Read the content once
                original_content = await response.aread()

                # Parse the streaming content
                complete_content = _parse_streaming_content(
                    original_content.decode('utf-8')
                )
                logger.info(
                    f"ASYNC RESPONSE ({response.status_code}) [COMPLETE STREAM]:\n{complete_content}"
                )

                # Return response with the content we read
                return httpx.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=original_content,
                    request=request,
                )
            else:
                # For non-streaming responses, log immediately
                content_bytes = await response.aread()
                content_str = content_bytes.decode("utf-8")

                try:
                    parsed_json = json.loads(content_str)
                    pretty_content = json.dumps(
                        parsed_json, indent=2, ensure_ascii=False
                    )
                    logger.info(
                        f"ASYNC RESPONSE ({response.status_code}):\n{pretty_content}"
                    )
                except json.JSONDecodeError:
                    logger.info(
                        f"ASYNC RESPONSE ({response.status_code}) [NON-JSON]:\n{content_str}"
                    )

                return httpx.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=content_bytes,
                    request=request,
                )
        except Exception as e:
            logger.warning(f"Failed to log async response body: {e}")
            return response


def log_streaming_response(
    content: str, url: str, status_code: int, is_async: bool = False
):
    """Helper function to log streaming responses after they've been consumed."""
    try:
        # Try to parse as JSON for pretty formatting
        try:
            parsed_json = json.loads(content)
            pretty_content = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            prefix = "ASYNC " if is_async else ""
            logger.info(
                f"{prefix}RESPONSE ({status_code}) [COMPLETE STREAM] to {url}:\n{pretty_content}"
            )
        except json.JSONDecodeError:
            # Log non-JSON responses as-is
            prefix = "ASYNC " if is_async else ""
            logger.info(
                f"{prefix}RESPONSE ({status_code}) [COMPLETE STREAM - NON-JSON] to {url}:\n{content}"
            )
    except Exception as e:
        logger.warning(f"Failed to log complete streamed response: {e}")
