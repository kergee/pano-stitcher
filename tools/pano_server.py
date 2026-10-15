"""Tauri sidecar entrypoint.

PyInstaller packages this file into pano-backend(.exe). The Tauri shell starts it
as a local FastAPI server, while the WebView loads app/static/index.html.
"""
from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="Pano local API server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run("app.main:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
