"""CPU-only real process cleanup check; never invokes the GPU benchmark."""
import json, os, sys, time
from pathlib import Path
import run_window as runner
root=Path(__file__).resolve().parent
receipt=root/'receipts/runner-process-smoke.json'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void',PYTHONDONTWRITEBYTECODE='1')
normal=runner.run_child([sys.executable,'-c','print("CPU_WORKER_OK")'],env=env,cwd=root,log_path=root/'receipts/worker-normal.log',timeout_seconds=10)
if normal['returncode']!=0 or normal['timed_out']:
    raise RuntimeError('Normal CPU child failed')
start=time.monotonic()
timeout=runner.run_child([sys.executable,'-c','import time; time.sleep(60)'],env=env,cwd=root,log_path=root/'receipts/worker-timeout.log',timeout_seconds=0.2)
elapsed=time.monotonic()-start
if not timeout['timed_out'] or elapsed>12 or timeout['returncode'] is None:
    raise RuntimeError('Owned CPU child timeout cleanup failed')
result=dict(status='passed',gpu_work_started=False,normal=normal,timeout=timeout,timeout_elapsed_seconds=elapsed)
receipt.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
