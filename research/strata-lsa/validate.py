#!/usr/bin/env python3
"""Verify this published evidence package without models, a GPU or network."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def main():
    checked = 0
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT), name
        assert path.is_file(), f'missing: {name}'
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert actual == digest, f'checksum mismatch: {name}'
        checked += 1
    checkpoint = json.loads((ROOT / 'native/CHECKPOINT.json').read_text())
    assert checkpoint['source_commit'] == '876307a9ded78bbca941189e63230b33cfdc6e2a'
    assert checkpoint['promotion'].startswith('not performed')
    map_file = ROOT / 'native/receipts/flashnext-learned-native-v1.bin'
    assert hashlib.sha256(map_file.read_bytes()).hexdigest() == checkpoint['map_sha256']
    assets = json.loads((ROOT / 'RELEASE-ASSETS.json').read_text())
    assert all(a['sha256'] and a['bytes'] > 0 for a in assets['assets'])
    print(f'PASS: {checked} published files; native map identity and no-promotion decision retained; {len(assets["assets"])} release assets indexed. No GPU work performed.')


if __name__ == '__main__':
    main()
