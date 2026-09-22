import argparse
import os
import subprocess
from pathlib import Path

import uvicorn

from .app import bridge_secret, create_app
from .config import Settings


def main():
    parser = argparse.ArgumentParser(description="Zelda AI Player — local agent laboratory")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--demo", action="store_true", help="Explicit simulator; not real Zelda gameplay")
    commands.add_parser("init")
    launch = commands.add_parser("launch-soh")
    launch.add_argument("executable", type=Path)
    args = parser.parse_args()
    settings = Settings()
    if args.command == "init":
        bridge_secret(settings)
        print("Local data directory and bridge credential initialized. No provider key is configured.")
    elif args.command == "launch-soh":
        if not args.executable.is_file():
            parser.error("SoH executable not found")
        environment = dict(os.environ)
        environment["ZELDA_BRIDGE_TOKEN"] = bridge_secret(settings)
        environment["ZELDA_BRIDGE_PORT"] = str(settings.bridge_port)
        subprocess.Popen([str(args.executable.resolve())], cwd=str(args.executable.resolve().parent), env=environment)
        print("SoH launched with the local bridge configuration. The executable must contain our native adapter.")
    else:
        settings.allow_simulator = args.demo
        uvicorn.run(create_app(settings), host="127.0.0.1", port=8787, access_log=False)


if __name__ == "__main__":
    main()
