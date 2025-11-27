#!/usr/bin/env python3
"""Copy core libfwht C sources into the Python package for sdist builds."""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
SRC_SRC = PROJECT_ROOT / "src"
INC_SRC = PROJECT_ROOT / "include"
DEST_ROOT = ROOT / "pyfwht" / "_libfwht"
DEST_SRC = DEST_ROOT / "src"
DEST_INC = DEST_ROOT / "include"

if not SRC_SRC.exists() or not INC_SRC.exists():
    print("Error: run this script from the python/ directory inside the repository.")
    sys.exit(1)

if DEST_ROOT.exists():
    shutil.rmtree(DEST_ROOT)

print(f"Vendoring libfwht sources into {DEST_ROOT} ...")
shutil.copytree(INC_SRC, DEST_INC)
shutil.copytree(SRC_SRC, DEST_SRC)
print("Done. New sources will be included in the next sdist build.")
