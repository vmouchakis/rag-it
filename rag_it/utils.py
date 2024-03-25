import os
from rag_it.config import INDICES_PATH


def _check_if_index_exists(index: str) -> bool:
    subfolder_path = os.path.join(INDICES_PATH, index)
    return os.path.isdir(subfolder_path)
