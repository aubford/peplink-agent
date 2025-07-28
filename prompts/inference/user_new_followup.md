# Task:
Reflect (silently) on the user query and the context corpus documents above and determine if they contain enough information to provide a comprehensive and accurate answer using only the knowledge in the context corpus. Then, either answer the question or execute an additional retrieval plan by following the retrieval procedure outlined above.

# Instructions:
- If you choose to answer the user's query, you may only use the knowledge in the context corpus documents.
- DO NOT use your own knowledge to answer the user's query.
- DO NOT reference the context corpus documents in your response.

# Output Format:
If you choose to answer, respond with JSON in the following format:
```json
{
  "have_enough_information": true,
  "have_enough_information_reasoning": "[REASONING]",
  "answer": "[ANSWER]"
}
```

If, on the other hand, you choose to execute an additional retrieval plan, make the appropriate tool calls.

# User Query:
"""
{user_query}
"""
