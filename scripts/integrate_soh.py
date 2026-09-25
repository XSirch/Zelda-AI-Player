"""Install controller-only bindings into the pinned, separate Shipwright checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

REVISION = "d30fc192f2eb01ceea45bd1e12de61636cafbf86"
PADMGR_BLOB = "173b03394bb47ebe317aba69c3a43ebf0cbb08b9"
INCLUDE = '#include "soh/Enhancements/zelda-ai/ZeldaAiBridge.h"\n'
LEGACY_CALL = "        ZeldaAiBridge_OverrideInput(i, &input->cur.button, &input->cur.stick_x, &input->cur.stick_y);\n"
CALL = "        ZeldaAiBridge_ConsumeInput(i, newInput, mode);\n"
ANCHOR = "        ogInput++;"
NATIVE_FILES = ("ZeldaAiBridge.cpp", "ZeldaAiBridge.h", "InputScheduler.hpp", "ActorRegistry.hpp")


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def patched_padmgr(raw: bytes) -> bytes:
    text = raw.decode("utf-8").replace("\r\n", "\n")
    # Undo only a complete, known patch and verify the entire reconstructed upstream.
    # This also catches local edits in a file that already contains our include/call.
    has_include = text.count(INCLUDE)
    old, new = text.count(LEGACY_CALL), text.count(CALL)
    if has_include or old or new:
        if has_include != 1 or old + new != 1:
            raise ValueError("Partial/duplicate adapter patch; inspect padmgr.c. No files were changed.")
        text = text.replace(INCLUDE, "", 1).replace(LEGACY_CALL if old else CALL, "", 1)
    if git_blob(text.encode()) != PADMGR_BLOB:
        raise ValueError("padmgr.c has unrecognized local changes; merge the consumer hook manually. No files were changed.")
    if text.count(ANCHOR) != 1 or "void PadMgr_RequestPadData(" not in text:
        raise ValueError("Input-consumer anchor is not unique.")
    return (INCLUDE + text.replace(ANCHOR, CALL + ANCHOR, 1)).encode()


def install(root: Path) -> Path:
    root = root.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected Shipwright {REVISION}; found {revision}. No reset was performed.")
    source = Path(__file__).resolve().parents[1] / "native"
    destination = root / "soh/soh/Enhancements/zelda-ai"
    padmgr = root / "soh/src/code/padmgr.c"
    # Read and validate every source before writing anything to the checkout.
    planned = {padmgr: patched_padmgr(padmgr.read_bytes())}
    planned.update({destination / name: (source / name).read_bytes() for name in NATIVE_FILES})
    manifest = {"bridge_build": "rt-input-v2.8", "protocol": 2, "upstream_revision": REVISION,
                "files": {p.name: hashlib.sha256(data).hexdigest() for p, data in planned.items()}}
    planned[destination / "installed-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    # Backups live OUTSIDE the source glob, so CMake cannot compile duplicate adapters.
    backup = root / ".zelda-ai-backups" / uuid.uuid4().hex
    originals = {path: path.read_bytes() if path.exists() else None for path in planned}
    backup.mkdir(parents=True)
    for path, content in originals.items():
        if content is not None:
            target = backup / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    written: list[Path] = []
    temporary: list[Path] = []
    try:
        for path, content in planned.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".zelda-stage-" + uuid.uuid4().hex)
            temporary.append(temp)
            temp.write_bytes(content)
            os.replace(temp, path)
            written.append(path)
    except Exception:
        for path in reversed(written):
            old = originals[path]
            if old is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(old)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    print(f"Adapter v2.8 installed. Previous files preserved in {backup}")
    print("Reconfigure and rebuild Shipwright; replacing source does not update an existing soh.exe.")
    print("Validate gameplay on Windows. This installer does not certify latency, navigation or combat.")
    return backup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("shipwright", type=Path)
    args = parser.parse_args()
    try:
        install(args.shipwright)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")
