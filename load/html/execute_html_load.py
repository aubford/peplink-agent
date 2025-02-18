from load.html.html_load import HTMLLoad

if __name__ == "__main__":
    loader = HTMLLoad()
    loader.load_from_merged()
    # Uncomment to load to vector store
    # loader.staging_to_vector_store()
