from langchain_core.rate_limiters import InMemoryRateLimiter


cohere_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.14,
    max_bucket_size=10,
)

openai_rate_limiter = InMemoryRateLimiter(
    requests_per_second=3.0,
    max_bucket_size=20,
)
