from langchain.docstore.document import Document
from pydantic import BaseModel
from load.base_load import BaseLoad
import pandas as pd
from load.batch_manager import BatchManager


class RedditData(BaseModel):
    id: str
    type: str
    primary_content: str
    title: str
    lead_content: str
    author_id: str
    author_name: str
    page_content: str
    score: int

    post_title: str
    post_content: str
    comment_content: str

    source_file: str
    subject_matter: str
    subreddit: str
    category: str
    url: str
    post_score: int
    comment_date: int
    post_id: str
    post_author_name: str
    post_author_id: str
    post_author_is_mod: bool
    post_author_is_gold: bool
    post_author_is_blocked: bool
    post_author_total_karma: str
    post_author_comment_karma: str
    post_author_link_karma: str
    post_author_verified: str
    post_author_has_verified_email: str
    post_author_has_subscribed: str
    post_author_is_employee: str
    comment_author_name: str
    comment_author_id: str
    comment_author_is_mod: bool
    comment_author_is_gold: bool
    comment_author_is_blocked: bool
    comment_author_total_karma: int
    comment_author_comment_karma: int
    comment_author_link_karma: int
    comment_author_verified: bool
    comment_author_has_verified_email: bool
    comment_author_has_subscribed: bool
    comment_author_is_employee: bool


class ModelResponse(BaseModel):
    technical_summary: str  # summary of post information
    is_useful: bool  # if false we will filter it out
    themes: list[str]  # themes of the post


class ForumSyntheticDataMixin:
    """
    Mixin class to generate prompts for OpenAI API to extract structured data from forum posts.
    The prompt facilitates transformation of a document's primary_content and lead_content
    into a structured format for analysis.
    """

    def create_system_prompt(self) -> str:
        """Creates the system prompt that instructs the model on its task."""
        return """You are an expert technical content analyzer specializing in IT networking and Pepwave products.
Your task is to analyze forum conversations and extract key information.

You will be provided with a forum conversation consisting of:
1. The original forum post/question.
2. A response to the original post.

Together, these form a single conversation turn between two forum users.

Analyze this conversation and provide a structured output containing:
1. A list of technical themes discussed.
2. A summary of the technical facts that can be gleaned from the conversation. Provide this in the form of sentences, not lists or other special formatting.
3. An assessment of whether there is useful technical information related to Pepwave products or IT networking.

Important guidelines:
- Focus only on technical content and information in your analysis.
- "Useful technical information" means factual statements, not questions asking for information.
- Be specific and precise in identifying themes.
- Provide a concise, factual summary that captures the key technical points.
- Base your analysis only on the provided content, do not make assumptions.
"""

    def create_prompt(self, lead_content: str, primary_content: str) -> str:
        """Creates a prompt that includes the forum post content."""

        return f"""# Forum Post (Original Post/Question)
{lead_content}

# Response to the Original Post
{primary_content}

Analyze this conversation according to the guidelines provided.
"""

    def get_examples(self) -> list[dict]:
        """
        Provides examples of conversations and ideal responses.

        Returns:
            List of example dictionaries.
        """
        return [
            {
                "lead_content": "I just installed a Pepwave MAX Transit Duo-CAT12 in my RV, but I'm having trouble with the cellular connection. The signal strength is showing only 2 bars even though my phone gets 4 bars in the same location. Does anyone know why this might be happening?",
                "primary_content": "Check your antenna connections first. The MAX Transit Duo requires proper external antennas to get the best signal. Make sure you're using the right cellular antennas and they're properly connected to the correct ports (they're labeled CELL on the router). Also, try changing the SIM priority in the admin panel - go to Network > Mobile > Settings, and you can change which SIM card is used or enable band locking for better performance on specific carriers. If you're in a fringe area, enabling band locking to the lower frequencies (like Band 12, 13, or 71 depending on your carrier) might help with penetration and range.",
                "expected_output": {
                    "themes": [
                        "cellular signal strength",
                        "antenna configuration",
                        "Pepwave MAX Transit Duo",
                        "SIM card settings",
                        "band locking",
                        "RV networking",
                    ],
                    "technical_summary": "To troubleshoot poor cellular signal on a Pepwave MAX Transit Duo-CAT12 when another device in the same location has good signal, it is recommended to check antenna connections to the correct ports, adjust SIM priority in the admin panel, and enable band locking for specific carriers, particularly lower frequency bands for better range in fringe areas.",
                    "is_useful": True,
                },
            },
            {
                "lead_content": "Anyone have recommendations for a good backup internet solution? I work from home and need something reliable when my main fiber connection goes down.",
                "primary_content": "I've been there! After trying several options, I settled on a Peplink Balance 20X with a 5G capable modem. The SpeedFusion technology in the Peplink devices is amazing for combining connections. I use it with both my fixed connection and a cellular backup, and the handover between them is completely seamless. I can be on a Zoom call and if my main connection fails, the call doesn't drop at all because of the Hot Failover feature. It's not cheap but worth every penny for reliability.",
                "expected_output": {
                    "themes": [
                        "backup internet solutions",
                        "Peplink Balance 20X",
                        "SpeedFusion technology",
                        "combining connections",
                        "Hot Failover",
                        "Work from home setup",
                        "videoconferencing",
                    ],
                    "technical_summary": "When in need of a backup internet solution for a work from home setup, a Peplink Balance 20X with a 5G capable modem is a good option. Its SpeedFusion technology is amazing for combining connections. The Hot Failover feature provides seamless transition between primary and backup connections, maintaining continuity for applications like video calls.",
                    "is_useful": True,
                },
            },
            {
                "lead_content": "Does anyone know if Peplink routers work with AT&T FirstNet? I'm looking to set up a mobile command center for our emergency response team.",
                "primary_content": "No idea, I've never used FirstNet. Have you tried contacting Peplink support directly? They might have better information about carrier compatibility.",
                "expected_output": {
                    "themes": [
                        "AT&T FirstNet compatibility",
                        "Peplink routers",
                        "emergency response equipment",
                    ],
                    "technical_summary": "To find out if Peplink routers work with AT&T FirstNet, one should contact Peplink support directly. The have better information about carrier compatibility.",
                    "is_useful": False,
                },
            },
        ]

    def create_system_prompt_with_examples(self) -> str:
        """
        Creates a system prompt that includes both instructions and examples.

        Returns:
            Complete system prompt with instructions and examples
        """
        base_prompt = self.create_system_prompt()
        examples = self.get_examples()

        examples_text = (
            "\n\nHere are some examples of conversations and expected analyses:\n\n"
        )

        for i, example in enumerate(examples, 1):
            # Format the example conversation
            example_prompt = self.create_prompt(
                lead_content=example["lead_content"],
                primary_content=example["primary_content"],
            )

            # Format the expected output
            output = example["expected_output"]
            themes_str = ", ".join([f'"{theme}"' for theme in output["themes"]])

            expected_output = (
                f'Expected output:\n'
                f'{{\n'
                f'  "themes": [{themes_str}],\n'
                f'  "technical_summary": "{output["technical_summary"]}",\n'
                f'  "is_useful": {output["is_useful"]}\n'
                f'}}\n'
            )

            examples_text += (
                f"EXAMPLE {i}:\n\n{example_prompt}\n\n{expected_output}\n{'=' * 40}\n\n"
            )

        return base_prompt + examples_text


class RedditLoad(BaseLoad, ForumSyntheticDataMixin):
    folder_name = "reddit"

    def __init__(self):
        super().__init__()
        self.batch_manager = BatchManager(self.staging_folder)
        self.nlp = self._init_spacy()

    def extract_entities(self, text: str) -> str:
        """Use SpaCy NER to extract entities from text, including custom PEPWAVE_SETTINGS_ENTITY."""
        if not text or pd.isna(text):
            return ""

        # Process the text with the pipeline that includes our EntityRuler
        doc = self.nlp(text)
        entities: set[str] = {
            ent.text
            for ent in doc.ents
            if ent.label_
            in {
                "URL",
                "FAC",
                "ORG",
                "GPE",
                "PRODUCT",
                "LOC",
                "WORK_OF_ART",
                "EVENT",
                "PEPWAVE_SETTINGS_ENTITY",  # Our custom entity type
            }
            and any(c.isalpha() for c in ent.text)
        }

        return ", ".join(entities) if entities else ""

    def ner(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use SpaCy NER feature to extract an "entities" column from the text in the primary_content and lead_content columns.
        The "entities" column contains a string that is a list of entities separated by commas.
        """

        # Process both columns and combine the entities
        df["primary_entities"] = df["comment_content"].apply(self.extract_entities)
        df["lead_entities"] = df["post_content"].apply(self.extract_entities)

        # Combine entities from both columns, removing duplicates
        def combine_entities(row):
            all_entities = []
            if row["primary_entities"]:
                all_entities.extend(row["primary_entities"].split(", "))
            if row["lead_entities"]:
                all_entities.extend(row["lead_entities"].split(", "))

            unique_entities = set(entity for entity in all_entities if entity)
            return ", ".join(unique_entities)

        df["entities"] = df.apply(combine_entities, axis=1)

        # Drop intermediate columns
        df = df.drop(columns=["primary_entities", "lead_entities"])

        return df

    def create_merged_df(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge all datasets into a single dataframe and perform operations."""
        df = pd.concat(dfs)
        df = self.ner(df)
        return df

    def load_docs(self, documents: list[Document]) -> list[Document]:
        """
        Process documents using the BatchManager and ForumSyntheticDataMixin.

        Args:
            documents: List of documents to process

        Returns:
            Processed documents with additional metadata
        """
        # Create batch items from documents
        batch_items = []
        for doc in documents:
            if not doc.id:
                raise ValueError("Document ID is required for batch processing")
            lead_content = doc.metadata.get("lead_content", "")
            primary_content = doc.metadata.get("primary_content", "")

            batch_items.append(
                {
                    "id": doc.id,
                    "prompt": self.create_prompt(lead_content, primary_content),
                }
            )

        # If we have batch items, create and run a batch job
        if batch_items:
            system_prompt = self.create_system_prompt_with_examples()
            self.batch_manager.create_batch_tasks(
                items=batch_items,
                schema=ModelResponse,
                system_prompt=system_prompt,
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=2040,
            )
            self.batch_manager.test_batchfile()

        return documents
