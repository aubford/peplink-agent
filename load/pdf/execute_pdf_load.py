from load.pdf.pdf_load import PDFLoad

if __name__ == "__main__":
    loader = PDFLoad()
    loader.load_from_merged()
    # Uncomment to load to vector store
    # loader.staging_to_vector_store()
