#!/usr/bin/env python3
"""Resume three pinned large wheels after uv transport resets; never use CUDA."""
import hashlib
import os
from pathlib import Path
import subprocess
import tomllib
from urllib.parse import urlparse

root = Path(__file__).resolve().parent
if root != Path('/path/to/lsa-prep-20261103'):
    raise SystemExit('Unexpected kit path')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NVIDIA_VISIBLE_DEVICES'] = 'void'
os.environ['UV_CACHE_DIR'] = str(root / 'cache/uv')
packages = {'nvidia-cublas-cu12', 'nvidia-cudnn-cu12', 'nvidia-nccl-cu12'}
lock = tomllib.loads((root / 'upstream/uv.lock').read_text())
wheelhouse = root / 'cache/wheels'
wheelhouse.mkdir(parents=True, exist_ok=True)
specs = []
for package in lock['package']:
    if package['name'] not in packages:
        continue
    choices = [w for w in package['wheels'] if 'manylinux' in w['url'] and 'x86_64' in w['url']]
    if len(choices) != 1:
        raise SystemExit('Ambiguous Linux wheel')
    wheel = choices[0]
    url = urlparse(wheel['url'])
    if url.scheme != 'https' or url.netloc != 'files.pythonhosted.org':
        raise SystemExit('Unexpected artifact host')
    dest = wheelhouse / Path(url.path).name
    expected = wheel['hash'].removeprefix('sha256:')
    def verified():
        if not dest.is_file() or dest.stat().st_size != wheel['size']:
            return False
        with dest.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest() == expected
    if not verified():
        if dest.exists() and dest.stat().st_size >= wheel['size']:
            raise SystemExit(f'Existing complete file failed verification: {dest.name}')
        print(f'Resuming {dest.name}', flush=True)
        subprocess.run(['curl', '--fail', '--location', '--http1.1', '--retry', '8', '--retry-all-errors', '--retry-delay', '2', '--continue-at', '-', '--max-time', '300', '--connect-timeout', '20', '--silent', '--show-error', '--output', str(dest), wheel['url']], check=True, timeout=900)
    if not verified():
        raise SystemExit(f'Artifact hash/size mismatch: {dest.name}')
    print(f'Hash verified {dest.name}', flush=True)
    specs.append(f"{package['name']}=={package['version']}")
if len(specs) != len(packages):
    raise SystemExit('Missing pinned package')
subprocess.run([str(root / '.bootstrap/bin/uv'), 'pip', 'install', '--python', str(root / '.venv/bin/python'), '--no-deps', '--no-index', '--find-links', str(wheelhouse), *specs], check=True)
print('Pinned wheels seeded; rerun prepare_env.sh to complete locked synchronization.', flush=True)
