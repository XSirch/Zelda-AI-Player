import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_walking_clearance_geometry(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("g++/clang++ unavailable; run the separate MSVC native harness on Windows")
    root = Path(__file__).parents[1]
    binary = tmp_path / "walking-clearance.exe"
    subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                    "-I", str(root / "native"), str(root / "tests/native_walking_clearance.cpp"),
                    "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=10)
