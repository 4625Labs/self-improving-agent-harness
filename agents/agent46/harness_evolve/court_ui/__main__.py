"""uv run python -m harness_evolve.court_ui --port 8765   (binds 127.0.0.1 only)"""

import argparse
import os
import sys

from .app import DEFAULT_DB, DEFAULT_URI, build_default_app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="harness_evolve.court_ui")
    p.add_argument("--uri", default=os.environ.get("EVOLVE_MONGODB_URI") or os.environ.get("MONGODB_URI") or DEFAULT_URI)
    p.add_argument("--db", default=os.environ.get("MONGODB_DATABASE", DEFAULT_DB))
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)
    import uvicorn

    app = build_default_app(a.uri, a.db)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
