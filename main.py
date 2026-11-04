"""Convenience entry point: ``python main.py`` == ``python -m edgegate``.

Adds ``src/`` to the path so the demos run from a fresh clone with no
installation step.  (For real use: ``pip install -e .`` then
``python -m edgegate`` works from anywhere.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from edgegate.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
