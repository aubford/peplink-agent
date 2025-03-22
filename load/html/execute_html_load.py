from load.html.html_load import HtmlLoad

if __name__ == "__main__":
    loader = HtmlLoad()
    loader.load()
    settings_entities = loader.get_all_settings_entities()
    print(settings_entities)
    # Uncomment to load to vector store
    # loader.staging_to_vector_store()
