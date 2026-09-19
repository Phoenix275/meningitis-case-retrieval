import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def records():
    with open(ROOT / "data.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="session")
def metadata():
    with open(ROOT / "metadata.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def index():
    faiss = pytest.importorskip("faiss")
    return faiss.read_index(str(ROOT / "index.faiss"))
