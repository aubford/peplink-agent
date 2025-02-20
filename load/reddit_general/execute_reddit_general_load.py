from load.reddit_general.reddit_general_load import RedditGeneralLoad

if __name__ == "__main__":
    loader = RedditGeneralLoad()
    loader.load()
    # Uncomment to load to vector store
    # loader.staging_to_vector_store()
