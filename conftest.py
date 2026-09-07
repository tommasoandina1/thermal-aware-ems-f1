"""
Make the repository root importable, whatever way pytest is invoked.

`python -m pytest` puts the current directory on sys.path; a bare `pytest`
does not. Since pytest 7 the rootdir is not added either, and because
tests/ has no __init__.py the only directory inserted is tests/ itself. The
suite therefore passes locally (where PYTHONPATH=/app is set by
docker-compose, or where `python -m pytest` is used) and fails in CI with
ModuleNotFoundError on `plant` and `rl` - a discrepancy that has nothing to
do with the code under test.

Putting this file at the repository root fixes the import path for every
invocation style, and pytest loads it automatically before collecting.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
