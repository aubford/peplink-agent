from rag_inference_langgraph import RagInferenceLangGraph
from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from langsmith import tracing_context
import uuid

load_dotenv()

# less verbose than debug
set_verbose(True)
# set_debug(True)

if __name__ == "__main__":
    with tracing_context(enabled=True, project_name="langchain-pepwave"):
        rag_inference = RagInferenceLangGraph(
            streaming=True,
            llm_model="gpt-4.1-mini",
            pinecone_index_name="pepwave-early-april-page-content-embedding",
        )

        # Generate a thread_id for this session to enable multi-turn memory
        thread_id = str(uuid.uuid4())

        while True:
            query = input("\n\n*** Enter a query (or 'exit' to exit): ")

            if query.lower() == "exit":
                break

            result = rag_inference.query(query, thread_id=thread_id)
            print(f"\n\nAssistant: {result['answer']}\n")
