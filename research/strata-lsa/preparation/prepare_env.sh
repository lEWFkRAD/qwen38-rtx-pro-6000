#!/usr/bin/env bash
set -euo pipefail
umask 077
KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
EXPECTED=/path/to/lsa-prep-20261103
[[ "$KIT" == "$EXPECTED" ]] || { printf 'Unexpected kit path: %s\n' "$KIT" >&2; exit 2; }
PIN=906be195c1052aaa74f95852ad1917ce1926ba64
export CUDA_VISIBLE_DEVICES=''
export NVIDIA_VISIBLE_DEVICES=void
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export UV_CACHE_DIR="$KIT/cache/uv"
export PIP_CACHE_DIR="$KIT/cache/pip"
export TRITON_CACHE_DIR="$KIT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$KIT/cache/torchinductor"
export UV_PROJECT_ENVIRONMENT="$KIT/.venv"
export PYTHONDONTWRITEBYTECODE=1
export UV_CONCURRENT_DOWNLOADS=2 UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=3
mkdir -p "$KIT/receipts" "$KIT/cache"
exec 9>"$KIT/prep.lock"
flock -n 9 || { printf 'Preparation already running\n' >&2; exit 3; }
if [[ ! -d "$KIT/upstream/.git" ]]; then
  git init -q "$KIT/upstream"
  git -C "$KIT/upstream" remote add origin https://github.com/Prefrontal-AI/latent-sparse-attention.git
  git -C "$KIT/upstream" fetch --depth 1 origin "$PIN"
  git -C "$KIT/upstream" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
fi
[[ "$(git -C "$KIT/upstream" rev-parse HEAD)" == "$PIN" ]] || { printf 'Source pin mismatch\n' >&2; exit 4; }
[[ -z "$(git -C "$KIT/upstream" status --porcelain --untracked-files=all)" ]] || { printf 'Upstream source changed\n' >&2; exit 4; }
if [[ ! -x "$KIT/.bootstrap/bin/python" ]]; then
  python3.12 -m venv "$KIT/.bootstrap"
fi
"$KIT/.bootstrap/bin/python" -m pip install --disable-pip-version-check --only-binary=:all: 'uv==0.13.0'
"$KIT/.bootstrap/bin/uv" --version > "$KIT/receipts/uv-version.txt"
cd "$KIT/upstream"
"$KIT/.bootstrap/bin/uv" sync --locked --python /usr/bin/python3.12 --no-managed-python
"$KIT/.bootstrap/bin/uv" pip freeze --python "$KIT/.venv/bin/python" > "$KIT/receipts/environment.freeze.txt"
"$KIT/.venv/bin/python" - <<'PY'
import importlib.metadata as md
import json, os, platform, sys
from pathlib import Path
import torch
root=Path(os.environ['UV_PROJECT_ENVIRONMENT']).parent
out={'python':sys.version,'platform':platform.platform(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'packages':{k:md.version(k) for k in ['torch','triton','nvidia-cutlass-dsl','apache-tvm-ffi','pytest','lsa']},'cuda_workload_performed':False}
arch_reader=getattr(torch._C, '_cuda_getArchFlags', None)
out['compiled_cuda_arch_flags']=arch_reader() if arch_reader else None
(root/'receipts/environment.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
PY
LSA_NO_SKIPS=1 "$KIT/.venv/bin/python" -m pytest -m 'not gpu and not cluster' tests/unit/core/test_reference.py tests/unit/core/test_ward.py tests/unit/core/test_attention.py tests/unit/core/test_index_space.py tests/unit/core/test_hierarchy.py tests/unit/core/test_dispatch.py --junitxml="$KIT/receipts/upstream-cpu-tests.xml" | tee "$KIT/receipts/upstream-cpu-tests.log"
git rev-parse HEAD > "$KIT/receipts/upstream-pin.txt"
git status --porcelain --untracked-files=all > "$KIT/receipts/upstream-status.txt"
printf 'CPU preparation complete; GPU qualification has NOT run.\n'
