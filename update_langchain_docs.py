import os
import shutil
import subprocess
from pathlib import Path

this_dir = Path(__file__).parent
docs_dest = this_dir / "langchain_docs"
langchain_dir = Path("/Users/aubrey/workspace/langchain")
tutorials_dir = Path("/Users/aubrey/workspace/_tutorials_and_examples")

print(
    "** Also consider updating langchain via `pip install --upgrade -r requirements.txt`"
)

if not os.path.exists(langchain_dir / "cookbook"):
    raise ValueError("Cookbook folder not found")


def copy_folders(src_dir: Path, dest_dir: Path, folders: list) -> None:
    for folder in folders:
        src_folder = src_dir / folder
        dest_folder = dest_dir / folder

        if os.path.exists(src_folder):
            print(f"Copying folder: {folder}")
            if os.path.exists(dest_folder):
                shutil.rmtree(dest_folder)  # Remove existing folder
            shutil.copytree(src_folder, dest_folder)
        else:
            print(f"Folder not found: {folder}")


folders_to_copy = [
    "concepts",
    "how_to",
    "integrations",
    "troubleshooting",
    "tutorials",
    "versions",
    "agents",
]

copy_folders(langchain_dir / "docs" / "docs", docs_dest, folders_to_copy)
copy_folders(langchain_dir, docs_dest / "cookbook", ["cookbook"])
copy_folders(
    tutorials_dir / "langgraph" / "docs" / "docs",
    docs_dest / "langgraph",
    folders_to_copy,
)

# Convert .ipynb files to .py
print("Converting .ipynb files to .py")
conversion_command = (
    f"find {docs_dest} "
    + r'-type f -name "*.ipynb" -exec jupyter nbconvert --to python {} \;'
)
subprocess.run(conversion_command, shell=True)

# Delete .ipynb files
print("Deleting .ipynb files")
delete_command = f'find {docs_dest} -type f -name "*.ipynb" -delete'
subprocess.run(delete_command, shell=True)
