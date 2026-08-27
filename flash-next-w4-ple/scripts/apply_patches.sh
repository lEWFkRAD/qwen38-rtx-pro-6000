#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/sglang-checkout" >&2
  exit 64
fi

for command_name in git python3; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "REFUSED: required command is unavailable: $command_name" >&2
    exit 69
  fi
done
if ! python3 -c 'import sys' >/dev/null 2>&1; then
  echo "REFUSED: python3 is present but cannot execute" >&2
  exit 69
fi

script_source=${BASH_SOURCE[0]}
script_parent=${script_source%/*}
if [[ "$script_parent" == "$script_source" ]]; then
  script_parent=.
fi
script_dir=$(cd -- "$script_parent" && pwd -P)
exec python3 "$script_dir/apply_patches_transaction.py" "$1"
