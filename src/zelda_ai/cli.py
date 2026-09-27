import argparse
import asyncio
import os
import subprocess
from pathlib import Path

import uvicorn

from .app import bridge_secret, create_app
from .config import Settings
from .providers import CodexProvider


async def auth_codex(settings: Settings):
    provider = CodexProvider(
        settings.codex_command,
        settings.codex_home,
        settings.decision_timeout_s,
    )
    try:
        status = await provider.status()
        if status.get("connected"):
            print("Codex/ChatGPT already authenticated in the Zelda AI Player profile.")
            return

        result = await provider.login(device=True)
        url = result.get("verificationUrl") or result.get("authUrl")
        code = result.get("userCode")
        if url:
            print(f"Open: {url}")
        if code:
            print(f"Code: {code}")
        print("Complete the ChatGPT sign-in in the browser. This command will detect completion.")

        for _ in range(180):
            await asyncio.sleep(2)
            status = await provider.status()
            if status.get("connected"):
                print("Codex/ChatGPT authenticated.")
                return
        raise RuntimeError("Codex login was not completed.")
    finally:
        await provider.close()


def main():
    parser = argparse.ArgumentParser(description="Zelda AI Player — autonomous ML player")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--demo", action="store_true", help="Explicit simulator; not real Zelda gameplay")
    commands.add_parser("init")
    commands.add_parser("auth-codex")
    launch = commands.add_parser("launch-soh")
    launch.add_argument("executable", type=Path)
    args = parser.parse_args()
    settings = Settings()

    if args.command == "init":
        bridge_secret(settings)
        print("Local data directory and bridge credential initialized.")
    elif args.command == "auth-codex":
        asyncio.run(auth_codex(settings))
    elif args.command == "launch-soh":
        if not args.executable.is_file():
            parser.error("SoH executable not found")
        environment = dict(os.environ)
        environment["ZELDA_BRIDGE_TOKEN"] = bridge_secret(settings)
        environment["ZELDA_BRIDGE_PORT"] = str(settings.bridge_port)
        subprocess.Popen(
            [str(args.executable.resolve())],
            cwd=str(args.executable.resolve().parent),
            env=environment,
        )
        print("SoH launched with the local bridge configuration. The executable must contain our native adapter.")
    else:
        settings.allow_simulator = args.demo
        uvicorn.run(
            create_app(settings),
            host="127.0.0.1",
            port=8787,
            access_log=False,
        )


if __name__ == "__main__":
    main()
