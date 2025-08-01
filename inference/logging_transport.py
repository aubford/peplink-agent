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
        lines = content.splitlines()
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
                    logger.warning(f"Failed to parse streaming line: {line}")
                    continue

        complete_content = ''.join(collected_chunks)

        if complete_content:
            # Return just the complete response content
            return json.dumps(
                json.loads(complete_content), indent=2, ensure_ascii=False
            )
        else:
            return "[No text (e.g. tool calls)]"

    except Exception as e:
        logger.warning(f"Failed to collect streaming content: {e}")
        return "[Failed to collect streaming content]"


class _TeeStream(httpx.SyncByteStream):
    """
    Pass-through stream that yields bytes to the caller while buffering them.
    When the caller finishes (close is called) we reconstruct the assistant
    content from the buffered chunks and write a single log entry.
    """

    def __init__(self, inner: httpx.SyncByteStream, url: str, status_code: int):
        self._inner = inner
        self._buf: list[bytes] = []
        self._url = url
        self._status = status_code

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._inner:
            self._buf.append(chunk)
            yield chunk

    def close(self) -> None:
        # Close the underlying stream first to release the connection
        self._inner.close()
        # Now assemble and log the full content reconstructed from SSE lines
        try:
            complete = _parse_streaming_content(
                b"".join(self._buf).decode("utf-8", errors="replace")
            )
            logger.info(
                f"RESPONSE ({self._status}) [COMPLETE STREAM] to {self._url}:\n{complete}"
            )
        except Exception as e:
            logger.warning(f"Failed to log complete streamed response: {e}")


class _AsyncTeeStream(httpx.AsyncByteStream):
    """
    Async counterpart to _TeeStream.
    """

    def __init__(self, inner: httpx.AsyncByteStream, url: str, status_code: int):
        self._inner = inner
        self._buf: list[bytes] = []
        self._url = url
        self._status = status_code

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._inner:
            self._buf.append(chunk)
            yield chunk

    async def aclose(self) -> None:
        await self._inner.aclose()
        try:
            complete = _parse_streaming_content(
                b"".join(self._buf).decode("utf-8", errors="replace")
            )
            logger.info(
                f"ASYNC RESPONSE ({self._status}) [COMPLETE STREAM] to {self._url}:\n{complete}"
            )
        except Exception as e:
            logger.warning(f"Failed to log complete streamed response: {e}")


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
                # Preserve true streaming semantics by teeing the underlying byte stream
                logger.info(
                    f"RESPONSE ({response.status_code}) [STREAMING]: Passing through stream for {request.url}"
                )
                tee = _TeeStream(
                    response.stream, str(request.url), response.status_code
                )
                return httpx.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    stream=tee,
                    request=request,
                    extensions=response.extensions,
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
                logger.info(
                    f"ASYNC RESPONSE ({response.status_code}) [STREAMING]: Passing through stream for {request.url}"
                )
                tee = _AsyncTeeStream(
                    response.stream, str(request.url), response.status_code
                )
                return httpx.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    stream=tee,
                    request=request,
                    extensions=response.extensions,
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
