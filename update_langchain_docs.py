import os
import shutil
import subprocess

print(
    "Also consider updating langchain via `pip install --upgrade -r requirements.txt`"
)


def copy_folders(src_dir: str, dest_dir: str, folders: list) -> None:
    for folder in folders:
        src_folder = os.path.join(src_dir, folder)
        dest_folder = os.path.join(dest_dir, folder)

        if os.path.exists(src_folder):
            print(f"Copying folder: {folder}")
            if os.path.exists(dest_folder):
                shutil.rmtree(dest_folder)  # Remove existing folder
            shutil.copytree(src_folder, dest_folder)
        else:
            print(f"Folder not found: {folder}")


source_dir = "/Users/aubrey/workspace/_tutorials/langchain/docs/docs"
destination_dir = "/Users/aubrey/workspace/langchain-pepwave/langchain_docs"
folders_to_copy = [
    "concepts",
    "cookbook",
    "how_to",
    "integrations",
    "troubleshooting",
    "tutorials",
    "versions",
]

copy_folders(source_dir, destination_dir, folders_to_copy)

cookbook_src = "/Users/aubrey/workspace/_tutorials/langchain/cookbook"
cookbook_dest = "/Users/aubrey/workspace/langchain-pepwave/langchain_docs/cookbook"

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
    f"find {destination_dir} "
    + r'-type f -name "*.ipynb" -exec jupyter nbconvert --to python {} \;'
)
subprocess.run(conversion_command, shell=True)

# Delete .ipynb files
print("Deleting .ipynb files")
delete_command = f'find {destination_dir} -type f -name "*.ipynb" -delete'
subprocess.run(delete_command, shell=True)
