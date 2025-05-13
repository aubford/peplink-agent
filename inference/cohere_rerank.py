from typing import List, Optional, Sequence
from langchain_cohere import CohereRerank
from langchain_core.callbacks import BaseCallbackManager, BaseCallbackHandler
from langchain_core.documents import Document
from inference.rate_limiters import cohere_rate_limiter


class RateLimitedCohereRerank(CohereRerank):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[List[BaseCallbackHandler] | BaseCallbackManager] = None,
    ) -> Sequence[Document]:
        """Compress documents using Cohere's rerank API with rate limiting."""
        if cohere_rate_limiter:
            cohere_rate_limiter.acquire()
        return super().compress_documents(documents, query, callbacks)

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[List[BaseCallbackHandler] | BaseCallbackManager] = None,
    ) -> Sequence[Document]:
        """Async compress documents with rate limiting."""
        if cohere_rate_limiter:
            await cohere_rate_limiter.aacquire()
        return await super().acompress_documents(documents, query, callbacks)
