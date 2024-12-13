import logging
from typing import Any, Dict
import pandas as pd
from fastparquet import write
from pathlib import Path


logger = logging.getLogger('transformer')

def save_parquet(data: Dict[str, Any], path: Path):
    # Convert to DataFrame and write to parquet
    df = pd.DataFrame([data])

    if not path.exists():
        write(path, df)
    else:
        write(path, df, append=True)


class BaseTransformer:
    """Base class for all data transformers."""
