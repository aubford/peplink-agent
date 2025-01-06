#%%
# from transform.base_transform import BaseTransform
import os
import sys
from pathlib import Path

print(f"Current working directory: {Path().resolve()}")
print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")
print(f"sys.path: {sys.path}")

# Try to find the util package
import pkgutil
print("\nLooking for util package:")
for finder, name, ispkg in pkgutil.iter_modules():
    if name.startswith('util'):
        print(f"Found: {name} (is package: {ispkg})")

# Import from the current directory
from util.util_main import test
test()
