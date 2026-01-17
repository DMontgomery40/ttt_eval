"""
Deprecated legacy dashboard.

This repo now has a single UI surface: the React dashboard under `dashboard/`
backed by the unified artifacts API in `ttt_ssm_nano.artifacts_api`.

Run:
  ./start.sh
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "ttt.ui.dashboard is deprecated. Run `./start.sh` to start the unified dashboard."
    )


if __name__ == "__main__":
    main()

