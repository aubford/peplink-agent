from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup
from extract.base_extractor import BaseExtractor, Ldoc
from util.util import serialize_document
import time


class WebsiteExtractor(BaseExtractor):
    source_name = "web"

    def __init__(self, url: str, *, base_url: str = None, max_depth: int = 6):
        super().__init__()
        self.file_id = (
            url.split("://", 1)[-1].replace("www.", "").replace("/", "_")[:40]
        )

        self.loader = RecursiveUrlLoader(
            url=url,
            base_url=base_url,
            use_async=False,
            max_depth=max_depth,
            extractor=self._extract_content,
            prevent_outside=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
            exclude_dirs=(
                "/cart",
                "/login",
                "/account",
                "/search",
                "/pdf",
                ".pdf",
                "/wp-admin/",
                "/wp-includes/",
                "/cgi-bin/",
                "/private/",
                "/tmp/",
                "/admin/",
                "/login/",
                "/user/",
                "/dashboard/",
                "/scripts/",
                "/styles/",
            ),
        )

    @staticmethod
    def _extract_content(html_content: str) -> str:
        """Extract clean text content from HTML"""
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove unwanted elements
        for element in soup.find_all(["nav", "footer", "script", "style"]):
            element.decompose()

        return soup.get_text(separator=" ", strip=True)

    def extract(self):
        """Extract and process web content"""
        stream_key = self.start_stream(Ldoc, identifier=self.file_id)
        for doc in self.loader.lazy_load():
            self.stream_item(serialize_document(doc), stream_key)
            time.sleep(1)  # Add 1 second delay between iterations
        self.end_stream(stream_key)
