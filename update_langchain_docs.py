import os
import shutil
import subprocess
from pathlib import Path

this_dir = Path(__file__).parent

print(
    "** Also consider updating langchain via `pip install --upgrade -r requirements.txt`"
)


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


tutorial_dir = Path("/Users/aubrey/workspace/_tutorials_and_examples/langchain")
docs_src = tutorial_dir / "docs" / "docs"
docs_dest = this_dir / "langchain_docs"
folders_to_copy = [
    "concepts",
    "cookbook",
    "how_to",
    "integrations",
    "troubleshooting",
    "tutorials",
    "versions",
]

copy_folders(docs_src, docs_dest, folders_to_copy)

cookbook_src = tutorial_dir / "cookbook"
cookbook_dest = docs_dest / "cookbook"

if os.path.exists(cookbook_src):
    print("Copying cookbook folder")
    if os.path.exists(cookbook_dest):
        shutil.rmtree(cookbook_dest)  # Remove existing cookbook folder
    shutil.copytree(cookbook_src, cookbook_dest)
else:
    print("Cookbook folder not found")

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
