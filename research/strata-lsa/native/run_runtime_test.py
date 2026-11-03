#!/usr/bin/env python3
"""Supervised standalone integrated-cache test; leaves production maintenance untouched."""
import contextlib, datetime, hashlib, json, os, signal, subprocess, time
from pathlib import Path
from test_integrated_engine import Guard, current_config
ROOT=Path(__file__).resolve().parent
OWNER='example-lsa-maintenance-owner'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    _,_,config_sha,contract_sha=current_config()
    guard=Guard(OWNER,config_sha,contract_sha,180)
    guard.check(force=True,before_start=True)
    directory=ROOT/'receipts'/('runtime-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    directory.mkdir(exist_ok=False)
    executable=ROOT/'build-engine/lsa_runtime_test'
    env={k:v for k,v in os.environ.items() if not k.startswith(('STRATA_','CUDA_','NVIDIA_','LD_'))}
    env.update(STRATA_LSA_MAPS=str(ROOT/'receipts/flashnext-learned-native-v1.bin'),CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH='/usr/local/cuda-13.0/lib64',OMP_NUM_THREADS='2')
    report={'status':'running','owner':OWNER,'executable_sha256':digest(executable),'script_sha256':digest(Path(__file__))}
    proc=None
    def stop(*_): raise KeyboardInterrupt('Runtime qualification interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGHUP,stop)
    try:
        with (directory/'worker.log').open('x') as log:
            proc=subprocess.Popen([str(executable)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                try: code=proc.wait(timeout=2);break
                except subprocess.TimeoutExpired: guard.check(force=True)
        guard.check(force=True)
        report.update(status='passed' if code==0 else 'failed',returncode=code)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):os.killpg(proc.pid,signal.SIGKILL)
                proc.wait(timeout=5)
        report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (directory/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'receipt':str(directory/'receipt.json')}),flush=True)
    return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
