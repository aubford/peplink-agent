from langchain_google_community import GoogleDriveLoader
from config import global_config

global_config.set("GOOGLE_APPLICATION_CREDENTIALS", "~/.credentials/credentials.json")

loader = GoogleDriveLoader(
    file_ids=["1tEBJ57pnw4Z-w40_JnbDVEoyfTZ8G0dr_-pvtteUdIQ"],
    # mime_types=[
    #     "application/vnd.google-apps.document",
    #     "application/vnd.google-apps.spreadsheet",
    #     "application/pdf"
    # ],
    # recursive=True
)

docs = loader.load()
print(f"Found {len(docs)} documents")
for doc in docs:
    print(f"- {doc.metadata.get('name', 'Unnamed document')}")
