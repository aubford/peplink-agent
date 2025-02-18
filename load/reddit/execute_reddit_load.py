from load.reddit.reddit_load import RedditLoad

if __name__ == "__main__":
    loader = RedditLoad()
    loader.load_from_merged()
    # Uncomment to load to vector store
    # loader.staging_to_vector_store()
