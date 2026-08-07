from __future__ import annotations

import argparse
import os

from dlubal.api import rfem


def main():
    parser = argparse.ArgumentParser(description="Check RFEM gRPC API connectivity.")
    parser.add_argument("--url", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--api-key", default=os.getenv("DLUBAL_API_KEY"))
    parser.add_argument("--api-key-name", default=os.getenv("DLUBAL_API_KEY_NAME"))
    args = parser.parse_args()

    with rfem.Application(
        api_key_value=args.api_key,
        api_key_name=args.api_key_name,
        url=args.url,
        port=args.port,
    ) as app:
        print(f"Connected to RFEM API at {args.url}:{args.port}")
        print(type(app).__name__)


if __name__ == "__main__":
    main()
