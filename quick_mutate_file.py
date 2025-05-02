import asyncio
import json
from pathlib import Path
import sys
from typing import Any
import pandas as pd
from load.base_load import BaseLoad


class Load(BaseLoad):
    folder_name = "foldername"

    def __init__(self):
        super().__init__()


async def mutate_data(data: Any) -> Any:
    df: pd.DataFrame = data
    loader = Load()
    df = loader._generate_embeddings(
        df, "page_content", chunk_size=200, clean_text=True
    )
    return df


async def main(file_path: Path) -> None:
    file = Path(file_path)
    ext = file.suffix.lower()
    data: list[dict]
    # Load data based on file extension
    try:
        if ext == ".json":
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                mutated_data = await mutate_data(data)
        elif ext == ".parquet":
            mutated_data = await mutate_data(pd.read_parquet(file))
        else:
            print(f"Unsupported file extension: {ext}")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        sys.exit(1)

    # Write a copy next to the original file with '_copy' appended before the extension
    try:
        copy_file = file.with_name(f"{file.stem}_copy{file.suffix}")
        if ext == ".json":
            with open(copy_file, "w", encoding="utf-8") as f:
                json.dump(
                    mutated_data.to_dict(orient="records"),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        elif ext == ".parquet":
            mutated_data.to_parquet(copy_file, index=False)
    except Exception as e:
        print(f"Error writing {copy_file}: {e}")
        sys.exit(1)

    print(mutated_data)


if __name__ == "__main__":
    root_dir = Path(__file__).parent
    asyncio.run(main(root_dir / "evals" / "output" / "kg_input_data.parquet"))
