import json
from pathlib import Path
from typing import List


def get_settings_entities() -> List[str]:
    """
    Load entities from settings_entities.json file

    Returns:
        List[str]: Entity names

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    json_path = Path(__file__).parent / 'settings_entities.json'

    with open(json_path, 'r') as f:
        entities = json.load(f)
        print(entities)
        return entities
