from typing import List, Dict, Any
from load.base_load import BaseLoad
import pandas as pd
import re
import json
from bs4 import BeautifulSoup
import os
from pathlib import Path


class HtmlLoad(BaseLoad):
    folder_name = "html"

    def __init__(self):
        super().__init__()

    def create_merged_df(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(dfs)
        # Add section as header in page_content.  Do this here instead of transform so we can experiment.
        mask = df["section"].str.strip().astype(bool)
        df.loc[mask, "page_content"] = (
            df.loc[mask, "section"] + "\n\n" + df.loc[mask, "page_content"]
        )
        df["images"] = df["images"].apply(list)
        df["settings_entities"] = df["page_content"].apply(self.get_entities_from_table)
        df["settings_entity_list"] = df["settings_entities"].apply(
            self.flatten_entities
        )

        # Create a frequency count for all entities across the dataframe
        all_entities = df["settings_entity_list"].explode().value_counts()
        # Keep only entities that appear exactly once in the entire dataframe
        unique_entities = all_entities[all_entities == 1].index.tolist()

        # Filter out entities that don't contain any letters (like "5 - 600")
        unique_entities = [
            entity for entity in unique_entities if any(c.isalpha() for c in entity)
        ]

        # Write unique entities to settings_entities.json
        unique_entities_path = self.staging_folder / "settings_entities.json"
        with open(unique_entities_path, 'w', encoding='utf-8') as f:
            json.dump(unique_entities, f, indent=2, ensure_ascii=False)
        self.logger.info(
            f"Saved {len(unique_entities)} unique settings entities to {unique_entities_path}"
        )

        # Filter each row's entity list to only include truly unique entities
        df["settings_entity_list"] = df["settings_entity_list"].apply(
            lambda x: [entity for entity in x if entity in unique_entities]
        )
        return df

    def clean_text(self, text: str) -> str:
        """
        Clean up text by removing excess whitespace and newlines.

        Args:
            text: The text to clean

        Returns:
            Cleaned text with normalized whitespace
        """
        # Replace newlines and multiple spaces with a single space
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned.strip()

    def flatten_entities(self, settings_entities: str) -> list[str]:
        """
        Extract keys from the JSON string of table entities.

        This function extracts all keys (including nested keys) from the settings
        entities JSON and returns them as a comma-separated string.

        Args:
            settings_entities: JSON string containing entity-description pairs

        Returns:
            Comma-separated string of entity keys found in the JSON
        """
        # Parse the JSON string into a Python dictionary
        if not settings_entities or settings_entities == "{}":
            return []

        data = json.loads(settings_entities)

        # Extract all keys (including nested keys) to process
        keys_to_process = []
        for key, value in data.items():
            keys_to_process.append(key)

            if isinstance(value, dict):
                # Add nested keys
                for nested_key in value.keys():
                    keys_to_process.append(nested_key)

        # Return comma-separated string of unique, sorted keys
        return sorted(set(keys_to_process)) if keys_to_process else []

    def get_entities_from_table(self, text: str) -> str:
        """
        Extract entity-description pairs from HTML tables in the content using BeautifulSoup.

        For each table:
        1. If the first row has only one cell, it's treated as the table title
        2. All subsequent rows with entity-description pairs are nested under that title
        3. Entity is extracted from the first <td> and description from subsequent <td>

        Note: Keys "Important Note" and "Tip" are excluded from the results.

        Args:
            text: The HTML content containing tables to be parsed

        Returns:
            A JSON string with proper formatting (double quotes, etc.) containing
            table titles as top-level keys and nested entity-description pairs as values
        """
        result = {}
        soup = BeautifulSoup(text, 'html.parser')
        tables = soup.find_all('table')

        # Keys to exclude
        excluded_keys = ["Important Note", "Tip", "Tips"]

        for table in tables:
            rows = table.find_all('tr')
            if not rows:
                continue

            table_title = None
            table_content = {}

            # Check if first row is a title (one cell only)
            first_row = rows[0]
            first_row_cells = first_row.find_all('td')

            if len(first_row_cells) == 1:
                table_title = self.clean_text(first_row_cells[0].get_text())
                # Skip the first row when processing entities
                entity_rows = rows[1:]

                # Skip tables with excluded titles
                if table_title in excluded_keys:
                    continue
            else:
                # No title, process all rows
                entity_rows = rows

            # Process entity-description pairs
            for row in entity_rows:
                cells = row.find_all('td')

                if len(cells) >= 2:
                    entity = self.clean_text(cells[0].get_text())
                    description = (
                        self.clean_text(cells[1].get_text()) if len(cells) > 1 else ""
                    )

                    # Skip excluded keys
                    if entity in excluded_keys:
                        continue

                    # Skip entities that are longer than 6 words
                    word_count = len(entity.split())
                    if word_count > 6:
                        continue

                    if entity:  # Only add if entity is not empty
                        if table_title:
                            # Add to nested dictionary under table title
                            table_content[entity] = description
                        else:
                            # No title, add directly to result
                            result[entity] = description

            # Add the table content to the result if there's a title and content
            if table_title and table_content:
                result[table_title] = table_content

        # Convert to a proper JSON string with double quotes
        try:
            json_str = json.dumps(result, ensure_ascii=False, indent=None)
            return json_str
        except (TypeError, ValueError) as e:
            print(f"Error serializing table entities: {e}")
            return "{}"

    @classmethod
    def get_all_settings_entities(cls) -> set[str]:
        """
        Get all unique settings_entities from the merged artifact.

        Returns:
            A set of all unique settings entities for use in nlp tasks.
        """
        df = cls.get_artifact(select_merged=True)
        # Explode the lists and filter out any NaN values (from empty lists)
        exploded = df["settings_entity_list"].explode()
        # Only keep actual string values (removes NaN from empty arrays)
        valid_entities = exploded[exploded.notna()]
        return set(valid_entities)
