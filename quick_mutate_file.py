import json
from pathlib import Path
import sys
import tempfile
import shutil


def mutate_data(data: list[dict]) -> list[dict]:
    return data


def main(file_path: str) -> None:
    file = Path(file_path)
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        sys.exit(1)

    mutated_data = mutate_data(data)

    # Write atomically
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
            json.dump(mutated_data, tf, indent=2, ensure_ascii=False)
            tempname = tf.name
        shutil.move(tempname, file)
    except Exception as e:
        print(f"Error writing {file}: {e}")
        sys.exit(1)

    print(mutated_data)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # Default path as in your original script
        root_dir = Path(__file__).parent.parent
        default_path = (
            root_dir
            / "testset-200_main_testset_25-04-23"
            / "generated_testset__TEST.json"
        )
        main(str(default_path))
