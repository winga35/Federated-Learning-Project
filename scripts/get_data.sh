#!/usr/bin/env bash
# Download CIFAR-10 into $FL_DATA_ROOT (default: ~/datasets/cifar10).
set -euo pipefail

ROOT="${FL_DATA_ROOT:-$HOME/datasets/cifar10}"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MD5="c58f30108f718f92721af3b95e74349a"
TARBALL="$ROOT/cifar-10-python.tar.gz"

if [ -d "$ROOT/cifar-10-batches-py" ]; then
    echo "already present: $ROOT/cifar-10-batches-py"
    exit 0
fi

mkdir -p "$ROOT"

echo "downloading -> $ROOT"
curl -fL --progress-bar -o "$TARBALL" "$URL"

echo "verifying checksum"
echo "$MD5  $TARBALL" | md5sum -c -

echo "extracting"
tar -xzf "$TARBALL" -C "$ROOT"

echo "done: $ROOT/cifar-10-batches-py"
