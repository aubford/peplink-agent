from langchain_google_community import GoogleDriveLoader
from config import global_config
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Set credentials
global_config.set(
    "GOOGLE_APPLICATION_CREDENTIALS", "/Users/aubrey/.credentials/credentials.json"
)

# Test direct API access
try:
    service = build("drive", "v3", cache_discovery=False)
    file_id = "1tEBJ57pnw4Z-w40_JnbDVEoyfTZ8G0dr_-pvtteUdIQ"

    # Try to get file metadata
    file = service.files().get(fileId=file_id, fields="name, mimeType").execute()
    print(f"Successfully accessed file: {file['name']} ({file['mimeType']})")

    # If we get here, try the loader
    loader = GoogleDriveLoader(file_ids=[file_id])

    docs = loader.load()
    print(f"\nFound {len(docs)} documents")
    for doc in docs:
        print(f"- {doc.metadata.get('name', 'Unnamed document')}")

except HttpError as error:
    print(f"Access Error: {error.reason}")
    print("Status code:", error.resp.status)
    if error.resp.status == 403:
        print(
            "Permission denied - check that your service account has access to this file"
        )
    elif error.resp.status == 404:
        print("File not found - check that the file ID is correct")
