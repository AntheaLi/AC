"""Wave 18e — trust audit CLI runner (compat shim).

Prefer `ac-trust-audit` (registered console script; same interface).
This shim delegates to `ac.cli_trust_audit.main` so historical
invocations (`python scripts/run_trust_audit.py ...`) keep working.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from ac.cli_trust_audit import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
