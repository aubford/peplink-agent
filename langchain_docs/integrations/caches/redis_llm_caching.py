#!/usr/bin/env python
# coding: utf-8

# # Redis Cache for LangChain
#
# This notebook demonstrates how to use the `RedisCache` and `RedisSemanticCache` classes from the langchain-redis package to implement caching for LLM responses.

# ## Setup
#
# First, let's install the required dependencies and ensure we have a Redis instance running.

# In[ ]:


get_ipython().run_line_magic(
    "pip", "install -U langchain-core langchain-redis langchain-openai redis"
)


# Ensure you have a Redis server running. You can start one using Docker with:
#
# ```
# docker run -d -p 6379:6379 redis:latest
# ```
#
# Or install and run Redis locally according to your operating system's instructions.

# In[2]:


import os

# Use the environment variable if set, otherwise default to localhost
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
print(f"Connecting to Redis at: {REDIS_URL}")


# ## Importing Required Libraries

# In[3]:


import time

from langchain.globals import set_llm_cache
from langchain.schema import Generation
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_redis import RedisCache, RedisSemanticCache


# In[4]:


import langchain_core
import langchain_openai
import openai
import redis


# ### Set OpenAI API key

# In[5]:


from getpass import getpass

# Check if OPENAI_API_KEY is already set in the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("OpenAI API key not found in environment variables.")
    openai_api_key = getpass("Please enter your OpenAI API key: ")

    # Set the API key for the current session
    os.environ["OPENAI_API_KEY"] = openai_api_key
    print("OpenAI API key has been set for this session.")
else:
    print("OpenAI API key found in environment variables.")


# ## Using RedisCache

# In[6]:


# Initialize RedisCache
redis_cache = RedisCache(redis_url=REDIS_URL)

# Set the cache for LangChain to use
set_llm_cache(redis_cache)

# Initialize the language model
llm = OpenAI(temperature=0)


# Function to measure execution time
def timed_completion(prompt):
    start_time = time.time()
    result = llm.invoke(prompt)
    end_time = time.time()
    return result, end_time - start_time


# First call (not cached)
prompt = "Explain the concept of caching in three sentences."
result1, time1 = timed_completion(prompt)
print(f"First call (not cached):\nResult: {result1}\nTime: {time1:.2f} seconds\n")

# Second call (should be cached)
result2, time2 = timed_completion(prompt)
print(f"Second call (cached):\nResult: {result2}\nTime: {time2:.2f} seconds\n")

print(f"Speed improvement: {time1 / time2:.2f}x faster")

# Clear the cache
redis_cache.clear()
print("Cache cleared")


# ## Using RedisSemanticCache

# In[7]:


# Initialize RedisSemanticCache
embeddings = OpenAIEmbeddings()
semantic_cache = RedisSemanticCache(
    redis_url=REDIS_URL, embeddings=embeddings, distance_threshold=0.2
)

# Set the cache for LangChain to use
set_llm_cache(semantic_cache)


# Function to test semantic cache
def test_semantic_cache(prompt):
    start_time = time.time()
    result = llm.invoke(prompt)
    end_time = time.time()
    return result, end_time - start_time


# Original query
original_prompt = "What is the capital of France?"
result1, time1 = test_semantic_cache(original_prompt)
print(
    f"Original query:\nPrompt: {original_prompt}\nResult: {result1}\nTime: {time1:.2f} seconds\n"
)

# Semantically similar query
similar_prompt = "Can you tell me the capital city of France?"
result2, time2 = test_semantic_cache(similar_prompt)
print(
    f"Similar query:\nPrompt: {similar_prompt}\nResult: {result2}\nTime: {time2:.2f} seconds\n"
)

print(f"Speed improvement: {time1 / time2:.2f}x faster")

# Clear the semantic cache
semantic_cache.clear()
print("Semantic cache cleared")


# ## Advanced Usage

# ### Custom TTL (Time-To-Live)

# In[8]:


# Initialize RedisCache with custom TTL
ttl_cache = RedisCache(redis_url=REDIS_URL, ttl=5)  # 60 seconds TTL

# Update a cache entry
ttl_cache.update("test_prompt", "test_llm", [Generation(text="Cached response")])

# Retrieve the cached entry
cached_result = ttl_cache.lookup("test_prompt", "test_llm")
print(f"Cached result: {cached_result[0].text if cached_result else 'Not found'}")

# Wait for TTL to expire
print("Waiting for TTL to expire...")
time.sleep(6)

# Try to retrieve the expired entry
expired_result = ttl_cache.lookup("test_prompt", "test_llm")
print(
    f"Result after TTL: {expired_result[0].text if expired_result else 'Not found (expired)'}"
)


# ### Customizing RedisSemanticCache

# In[9]:


# Initialize RedisSemanticCache with custom settings
custom_semantic_cache = RedisSemanticCache(
    redis_url=REDIS_URL,
    embeddings=embeddings,
    distance_threshold=0.1,  # Stricter similarity threshold
    ttl=3600,  # 1 hour TTL
    name="custom_cache",  # Custom cache name
)

# Test the custom semantic cache
set_llm_cache(custom_semantic_cache)

test_prompt = "What's the largest planet in our solar system?"
result, _ = test_semantic_cache(test_prompt)
print(f"Original result: {result}")

# Try a slightly different query
similar_test_prompt = "Which planet is the biggest in the solar system?"
similar_result, _ = test_semantic_cache(similar_test_prompt)
print(f"Similar query result: {similar_result}")

# Clean up
custom_semantic_cache.clear()


# ## Conclusion
#
# This notebook demonstrated the usage of `RedisCache` and `RedisSemanticCache` from the langchain-redis package. These caching mechanisms can significantly improve the performance of LLM-based applications by reducing redundant API calls and leveraging semantic similarity for intelligent caching. The Redis-based implementation provides a fast, scalable, and flexible solution for caching in distributed systems.
