"""Make `python -m pytest tests/` work from a plain checkout.

CI only ever passed because it runs `pip install -e ".[all]"` first, which puts
both the repo root and `src/` on sys.path as a side effect. Without that editable
install, fourteen test modules failed to COLLECT:

    ModuleNotFoundError: No module named 'enzkin'            (the src/ package)
    ModuleNotFoundError: No module named 'scripts.qsar_congeneric'
                                                            (the scripts/ package)

So a reader who cloned the repo and ran the documented command got an error wall
instead of the suite. Both paths are added here, which is test-harness setup, not
a change to any test or assertion.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
