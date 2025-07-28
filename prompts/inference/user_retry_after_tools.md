The documents retrieved in the prior tool calls have been merged into the context corpus above. Reflect (silently) on the user query and the context corpus documents and determine if it now contains enough information to provide a comprehensive and accurate answer to the user's query. Then, either answer the query or execute an additional retrieval plan by following the retrieval procedure outlined above.

# Instructions:
- If you choose to answer the user's query, you may only use the knowledge in the context corpus documents.
- DO NOT use your own knowledge to answer the user's query.
- DO NOT reference the context corpus documents in your response.
- If you choose to execute an additional retrieval plan, make sure not to repeat queries that have already been performed.

# User Query:
"""
{user_query}
"""
