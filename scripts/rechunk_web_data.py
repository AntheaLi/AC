"""Rebuild the v1-web chunked JS payloads from compiler-data.json.

Wave 30 (Jul 2026): companion to emit_decision_grid.py --rebuild. Loads
the multi-hardware payload, re-runs the post-search pipeline (family
rollup / smoothing / canonical-shape pin / plateau markers — no optimizer
searches), rewrites compiler-data.json, and re-chunks it into the
compiler-data-00N.js files the demo's index.html loads via
window.__AC_DATA_CHUNKS__.

Usage:
  python3 scripts/rechunk_web_data.py WEB_DIR [--n-chunks 5] [--no-rebuild]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "ac"), _ROOT, _HERE, os.path.join(_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _escape_template_literal(s: str) -> str:
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("web_dir")
    ap.add_argument("--n-chunks", type=int, default=5)
    ap.add_argument("--no-rebuild", action="store_true",
                    help="skip the rollup chain; just re-chunk")
    args = ap.parse_args()

    json_path = os.path.join(args.web_dir, "compiler-data.json")
    with open(json_path) as f:
        data = json.load(f)

    if not args.no_rebuild:
        from emit_decision_grid import rebuild_chain
        data = rebuild_chain(data)
        with open(json_path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"Rebuilt rollup chain and rewrote {json_path}")

    # Split the RAW compact JSON, then escape each chunk independently.
    # (Escaping first and then splitting could cut an escape sequence in
    # half at a chunk boundary, leaving a dangling backslash that escapes
    # the chunk's own closing backtick — a JS syntax error.)
    compact = json.dumps(data, separators=(",", ":"))
    n = max(1, args.n_chunks)
    size = (len(compact) + n - 1) // n
    for i in range(n):
        chunk = _escape_template_literal(compact[i * size:(i + 1) * size])
        path = os.path.join(args.web_dir, f"compiler-data-{i:03d}.js")
        with open(path, "w") as f:
            f.write("window.__AC_DATA_CHUNKS__ = "
                    "window.__AC_DATA_CHUNKS__ || [];\n")
            f.write(f"window.__AC_DATA_CHUNKS__.push(`{chunk}`);\n")
        print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
