"""Compatibility wrapper for running the G4 paper scenario.

G4 is handled by the unified paper-group runner. This file preserves the old
entry point while delegating to ``run_paper_extra_groups.py --groups g4``.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_paper_extra_groups import main as _unified_main


def _has_groups_arg(argv):
    return any(arg == "--groups" or arg.startswith("--groups=") for arg in argv)


if __name__ == "__main__":
    if not _has_groups_arg(sys.argv[1:]):
        sys.argv.extend(["--groups", "g4"])
    raise SystemExit(_unified_main())
