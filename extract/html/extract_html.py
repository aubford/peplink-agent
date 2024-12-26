from extract.html.html_text_splitters import HTMLSemanticPreservingSplitter

def h3_pagetitle_filter(element):
    return element.name == "h3" and "inline-pagetitle" in element.get("class", [])

# Configure splitter
splitter = HTMLSemanticPreservingSplitter(
    max_chunk_size=3000,
    headers_to_split_on=[("h3", "Section")],
    elements_to_preserve=["table","ul","ol"],
    tags_to_preserve=["table", "tr", "td", "th","li"],
    preserve_image_metadata=True
)

# Load and split the HTML file
with open("manual.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# Split into chunks
chunks = splitter.split_text(html_content)

# Print each chunk separated by a divider
for chunk in chunks:
    print("Content: \n", chunk.page_content)
    print("\n")
    print("Metadata: \n", chunk.metadata)
    print("------------\n")
