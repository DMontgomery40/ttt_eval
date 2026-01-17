from __future__ import annotations

import argparse

import uvicorn

from .app import create_app


def main() -> None:
    p = argparse.ArgumentParser(description="Serve Phase 1 artifacts to the React dashboard")
    p.add_argument("--artifacts_root", type=str, default="artifacts", help="Artifacts root directory")
    p.add_argument("--host", type=str, default="127.0.0.1")
    # Use a less common port by default (8000 is frequently occupied)
    p.add_argument("--port", type=int, default=13579)
    args = p.parse_args()

    app = create_app(artifacts_root=args.artifacts_root)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
