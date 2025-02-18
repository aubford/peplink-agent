from load.web.web_load import WebLoad

if __name__ == "__main__":
    loader = WebLoad()
    loader.load_from_merged()
    # Uncomment to load to vector store
    # loader.staging_to_vector_store()
