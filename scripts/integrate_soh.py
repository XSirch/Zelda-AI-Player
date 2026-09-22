"""Install our source adapter into a separate, pinned Shipwright checkout. Never copies ROMs."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

REVISION = "d30fc192f2eb01ceea45bd1e12de61636cafbf86"
PADMGR_BLOB = "173b03394bb47ebe317aba69c3a43ebf0cbb08b9"
INCLUDE = '#include "soh/Enhancements/zelda-ai/ZeldaAiBridge.h"\n'
CALL = "        ZeldaAiBridge_OverrideInput(i, &input->cur.button, &input->cur.stick_x, &input->cur.stick_y);\n"
ANCHOR = "        buttonDiff = input->prev.button ^ input->cur.button;"


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def patched_padmgr(raw: bytes) -> bytes:
    text = raw.decode("utf-8").replace("\r\n", "\n")
    if INCLUDE in text and CALL in text:
        return raw
    if INCLUDE in text or CALL in text:
        raise ValueError("Partial adapter patch found; inspect padmgr.c before proceeding.")
    normalized = text.encode()
    if git_blob(normalized) != PADMGR_BLOB:
        raise ValueError("padmgr.c differs from the verified upstream file; no files were changed.")
    if text.count(ANCHOR) != 1:
        raise ValueError("Input integration anchor is not unique.")
    return (INCLUDE + text.replace(ANCHOR, CALL + ANCHOR, 1)).encode()


def install(root: Path):
    root = root.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected Shipwright {REVISION}; found {revision}. Use a separate pinned checkout.")
    padmgr = root / "soh/src/code/padmgr.c"
    patched = patched_padmgr(padmgr.read_bytes())
    source = Path(__file__).resolve().parents[1] / "native"
    destination = root / "soh/soh/Enhancements/zelda-ai"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("ZeldaAiBridge.cpp", "ZeldaAiBridge.h", "InputLease.hpp"):
        shutil.copy2(source / name, destination / name)
    padmgr.write_bytes(patched)
    print("Adapter source installed. Reconfigure and build Shipwright using its Windows build instructions.")
    print("The source bindings still require an actual SoH build and in-game validation on your PC.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("shipwright", type=Path)
    args = parser.parse_args()
    try:
        install(args.shipwright)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")
