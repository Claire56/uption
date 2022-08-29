import os 
from pathlib import Path

ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)

def get_project_root() -> Path:
    return Path(__file__).parent

def get_project_root_old() -> Path:
    return Path(__file__).parent.parent
print(get_project_root)