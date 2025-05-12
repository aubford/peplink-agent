from rag_inference import RagInference
from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from langsmith import tracing_context

load_dotenv()

# less verbose than debug
set_verbose(True)
# set_debug(True)

if __name__ == "__main__":
    with tracing_context(enabled=True, project_name="langchain-pepwave"):
        rag_inference = RagInference(
            streaming=True,
            llm_model="gpt-4.1-mini",
            pinecone_index_name="pepwave-early-april-page-content-embedding",
        )

        while True:
            query = input("\n\n*** Enter a query (or 'exit' to exit): ")

            if query.lower() == "exit":
                break

            result = rag_inference.query(query)
            print(f"\n\nAssistant: {result['answer']}\n")
