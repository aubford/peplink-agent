from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import global_config
from langchain import hub
from pinecone import Pinecone
from langchain_core.runnables.passthrough import RunnablePassthrough
from inference.history_aware_retrieval_query import history_aware_retrieval_query

# Note: for reasoning models: "include only the most relevant information to prevent the model from overcomplicating its response." - api docs
# Other advice for reasoning models: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting

pinecone = Pinecone(api_key=global_config.get("PINECONE_API_KEY"))
# don't forget to update index to new version
index = pinecone.Index("pepwave")
embeddings = OpenAIEmbeddings()
# note: will need to remove the namespace for future indexes
vector_store = PineconeVectorStore(
    index=index, embedding=embeddings, text_key="text", namespace="pepwave"
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
prompt = hub.pull("aubford/retrieval-qa-chat")

retriever = (lambda x: x["retrieval_query"]) | vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 20, "fetch_k": 50}
)

retrieval_chain = (
    RunnablePassthrough.assign(retrieval_query=history_aware_retrieval_query())
    .assign(
        context=retriever.with_config(run_name="retrieve_documents"),
    )
    .assign(answer=create_stuff_documents_chain(llm, prompt))
).with_config(run_name="rag_inference")


if __name__ == "__main__":
    chat_history = []
    while True:
        # Get user input
        query = input("\n\n*** Enter a query (or 'exit' to exit): ")

        # Check for exit condition
        if query.lower() == "exit":
            break

        result = retrieval_chain.invoke({"input": query, "chat_history": chat_history})
        print(f"\n\nAssistant: {result['answer']}\n")
        chat_history.append(("human", query))
        chat_history.append(("assistant", result["answer"]))
