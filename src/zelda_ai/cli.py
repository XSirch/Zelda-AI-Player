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
    qualify = commands.add_parser("qualify-motor", help="Real SoH motor episode without any AI calls")
    qualify.add_argument("--seconds", type=float, default=120.0)
    qualify.add_argument("--wait-seconds", type=float, default=8.0)
    qualify.add_argument("--probe-only", action="store_true")
    qualify.add_argument("--save-slot", type=int, choices=(1, 2, 3), help="Explicit existing save to load through physical input")
    g1 = commands.add_parser("qualify-g1", help="Automatic real SoH G1 batch; no AI calls or training")
    g1.add_argument("executable", type=Path)
    g1.add_argument("--source-home", type=Path, required=True, help="Existing native config/Save directory, copied locally")
    g1.add_argument("--save-slot", type=int, choices=(1, 2, 3), required=True)
    g1.add_argument("--episodes", type=int, default=100, help="100 for qualification; smaller batches are pilots")
    g1.add_argument("--seed", type=int, default=1042026)
    g1.add_argument("--seconds", type=float, default=120.0, help="Complete scenario budget, including startup and setup")
    surfaces = commands.add_parser("train-local-surfaces", help="Isolated local imitation and real before/after trials; zero AI calls")
    surfaces.add_argument("executable", type=Path)
    surfaces.add_argument("--source-home", type=Path, required=True)
    surfaces.add_argument("--save-slot", type=int, choices=(1, 2, 3), required=True)
    surfaces.add_argument("--demonstrations", type=int, default=24)
    surfaces.add_argument("--evaluation", type=int, default=12, help="Divisible by 3; 300 tests 100 per family")
    surfaces.add_argument("--retention", type=int, default=10)
    surfaces.add_argument("--seed", type=int, default=4102026)
    launch = commands.add_parser("launch-soh")
    launch.add_argument("executable", type=Path)
    args = parser.parse_args()
    settings = Settings()

    if args.command == "train-local-surfaces":
        from .surface_curriculum import surface_curriculum
        report = asyncio.run(surface_curriculum(settings, args.executable, args.source_home,
            save_slot=args.save_slot, demonstration_trials=args.demonstrations,
            evaluation_trials=args.evaluation, retention_trials=args.retention, seed=args.seed))
        if not report["artifacts_unchanged"]:
            raise SystemExit(2)
    elif args.command == "qualify-g1":
        if not 1 <= args.episodes <= 100 or not 10 <= args.seconds <= 120:
            parser.error("G1 allows 1..100 episodes and a complete-scenario budget of 10..120 seconds")
        from .g1 import qualify_g1
        report = asyncio.run(qualify_g1(settings, args.executable, args.source_home,
            episodes=args.episodes, seed=args.seed, save_slot=args.save_slot, seconds=args.seconds))
        if not report["g1_qualified"]:
            raise SystemExit(2)
    elif args.command == "qualify-motor":
        if not 1 <= args.seconds <= 600 or not 1 <= args.wait_seconds <= 60:
            parser.error("Qualification budgets must be 1..600 seconds and wait 1..60 seconds")
        from .qualification import qualify_motor
        report = asyncio.run(qualify_motor(settings, seconds=args.seconds,
            wait_s=args.wait_seconds, probe_only=args.probe_only, save_slot=args.save_slot))
        if report["status"] == "blocked":
            raise SystemExit(2)
    elif args.command == "init":
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
