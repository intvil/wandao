#!/usr/bin/env python3
"""
Run the draft training pipeline.

Can be invoked as a module (`python -m draft.cli.run_draft`) or as a script
(`python cli/run_draft.py`) from the repo root.
"""

import os
import sys

try:
    from draft.pipelines.main import main
except ImportError:
    # Allow running as a script from repo root without installing the package
    PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PROJECT_ROOT = os.path.abspath(os.path.join(PKG_ROOT, ".."))
    for path in (PROJECT_ROOT,):
        if path not in sys.path:
            sys.path.insert(0, path)
    from draft.pipelines.main import main


if __name__ == "__main__":
    main()
