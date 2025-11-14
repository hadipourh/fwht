#!/bin/bash
#
# Version Management Script for libfwht
# 
# Usage: ./tools/update_version.sh <new_version>
# Example: ./tools/update_version.sh 1.2.1
#
# This script updates version numbers in all relevant files from a single source.

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.2.1"
    exit 1
fi

NEW_VERSION="$1"

# Validate version format (X.Y.Z)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y.Z (e.g., 1.2.1)"
    exit 1
fi

# Extract major, minor, patch
MAJOR=$(echo "$NEW_VERSION" | cut -d. -f1)
MINOR=$(echo "$NEW_VERSION" | cut -d. -f2)
PATCH=$(echo "$NEW_VERSION" | cut -d. -f3)

echo "Updating version to $NEW_VERSION (major=$MAJOR, minor=$MINOR, patch=$PATCH)..."

# Update C library header
echo "  → include/fwht.h"
sed -i.bak \
    -e "s/^#define FWHT_VERSION_MAJOR .*/#define FWHT_VERSION_MAJOR $MAJOR/" \
    -e "s/^#define FWHT_VERSION_MINOR .*/#define FWHT_VERSION_MINOR $MINOR/" \
    -e "s/^#define FWHT_VERSION_PATCH .*/#define FWHT_VERSION_PATCH $PATCH/" \
    -e "s/^#define FWHT_VERSION .*/#define FWHT_VERSION \"$NEW_VERSION\"/" \
    -e "s/^ \* Version: .*/ * Version: $NEW_VERSION/" \
    include/fwht.h
rm include/fwht.h.bak

# Update Python package version
echo "  → python/pyfwht/_version.py"
sed -i.bak "s/__version__ = .*/__version__ = \"$NEW_VERSION\"/" python/pyfwht/_version.py
rm python/pyfwht/_version.py.bak

# Update Python pyproject.toml
echo "  → python/pyproject.toml"
sed -i.bak "s/^version = .*/version = \"$NEW_VERSION\"/" python/pyproject.toml
rm python/pyproject.toml.bak

# Update test file
echo "  → tests/test_correctness.c"
sed -i.bak "s/strcmp(version, \".*\")/strcmp(version, \"$NEW_VERSION\")/" tests/test_correctness.c
rm tests/test_correctness.c.bak

# Sync to Python package
echo ""
echo "Running sync script to update Python package sources..."
bash tools/sync_python.sh

echo ""
echo "✓ Version updated to $NEW_VERSION in all files"
echo ""
echo "Next steps:"
echo "  1. Run 'make clean && make' to verify build"
echo "  2. Commit changes: git commit -am \"Bump version to $NEW_VERSION\""
echo "  3. Tag release: git tag -a v$NEW_VERSION -m \"Release v$NEW_VERSION\""
echo "  4. Push: git push origin main --tags"
