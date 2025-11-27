#!/usr/bin/env bash
set -euo pipefail

# Automates the full pyfwht release workflow.
# Usage: ./python/tools/run_release.sh X.Y.Z

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <version>" >&2
  exit 1
fi

VERSION="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
VENV_PIP="$VENV_DIR/bin/pip"
VENV_PY="$VENV_DIR/bin/python"
VENV_TWINE="$VENV_DIR/bin/twine"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: expected virtual environment python at $VENV_PY" >&2
  exit 1
fi
if [[ ! -x "$VENV_PIP" ]]; then
  echo "Error: expected pip at $VENV_PIP" >&2
  exit 1
fi

log() {
  printf '\n==> %s\n' "$1"
}

log "Upgrading packaging utilities"
"$VENV_PIP" install --upgrade pip build twine

cd "$REPO_ROOT"

log "Updating version to $VERSION"
./tools/update_version.sh "$VERSION"

log "Building and testing C library"
make clean
make test

log "Running Python tests"
pushd python >/dev/null
"$VENV_PIP" install -e .
"$VENV_PY" -m pytest tests -v
popd >/dev/null

log "Preparing source distribution"
pushd python >/dev/null
rm -rf dist build *.egg-info
"$VENV_PY" tools/vendor_libfwht_sources.py
"$VENV_PY" -m build --sdist
popd >/dev/null

log "Committing and tagging"
git add -A
git commit -m "Release v$VERSION"
git tag "v$VERSION"
git push origin main --tags

log "Uploading to PyPI"
pushd python >/dev/null
"$VENV_TWINE" upload dist/*
popd >/dev/null

log "Release v$VERSION complete"
