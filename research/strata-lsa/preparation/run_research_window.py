"""Manually run the LSA diagnostic during a coordinated GPU window.
Default is a plan. --check-window is read-only. --execute never changes services.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
ORIGINAL_PIN = '906be195c1052aaa74f95852ad1917ce1926ba64'
PIN = '35b1bf9a10703b57da3eb073debfad6a66507269'
PRIMARY = 'qwen38-flash-next-strata.service'
CONTEXTS = (131072, 262144, 1048576)
class Refusal(RuntimeError):
    pass

def utc():
    return datetime.now(timezone.utc).isoformat()

def within(root, path):
    root, resolved = Path(root).resolve(), Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise Refusal(f'Path escapes the experiment kit: {path}')
    return resolved

def command(argv, timeout=15):
    try:
        r = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Refusal(f'Required read failed: {argv[0]}: {exc}') from exc
    if r.returncode:
        raise Refusal(f'Required read failed ({r.returncode}): {argv[0]}: {r.stderr.strip()[:800]}')
    return r.stdout.strip()

def primary_state(timeout=5):
    raw=command(['systemctl','show',PRIMARY,'--property=LoadState','--property=ActiveState'],timeout=timeout)
    fields=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
    if fields.get('LoadState') != 'loaded':
        raise Refusal('Primary service definition is not loaded')
    return fields.get('ActiveState','')

def gpu_snapshot():
    raw = command(['nvidia-smi','--id=0','--query-gpu=uuid,name,compute_cap,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'])
    try:
        rows = list(csv.reader(io.StringIO(raw), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != 6:
            raise ValueError('expected one six-column GPU row')
        identity,name,capability,total,free,util = rows[0]
        item = dict(uuid=identity,name=name,capability=capability,total_mib=int(total),free_mib=int(free),utilization_percent=int(util),sampled_at=utc())
        if not identity or not 0 <= item['utilization_percent'] <= 100:
            raise ValueError('invalid GPU metadata')
        if not 0 <= item['free_mib'] <= item['total_mib']:
            raise ValueError('invalid GPU memory counters')
        return item
    except (ValueError,TypeError) as exc:
        raise Refusal(f'Cannot establish GPU state: {exc}') from exc

def check_window(budget_mib, sample_count=3, pause_seconds=2):
    state = primary_state()
    if state != 'inactive':
        raise Refusal(f'{PRIMARY} must already be inactive; observed {state!r}. Launcher never stops it or changes recovery.')
    snapshots=[]
    for index in range(sample_count):
        item=gpu_snapshot()
        if item['capability'] != '12.0':
            raise Refusal(f"Expected compute capability12.0, got {item['capability']}")
        if snapshots and item['uuid'] != snapshots[0]['uuid']:
            raise Refusal('GPU identity changed during preflight')
        minimum=max(8192,budget_mib+4096)
        if item['free_mib'] < minimum:
            raise Refusal(f"Need {minimum}MiB free; got {item['free_mib']}")
        if item['utilization_percent'] > 5:
            raise Refusal(f"GPU busy ({item['utilization_percent']}%)")
        snapshots.append(item)
        if index+1 < sample_count:
            time.sleep(pause_seconds)
    if primary_state() != 'inactive':
        raise Refusal('Primary state changed during preflight')
    return snapshots

def hash_file(path):
    with open(path,'rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def source_provenance(root):
    repo=within(root,root/'upstream-research')
    pin=command(['git','-C',str(repo),'rev-parse','HEAD'])
    if pin != PIN:
        raise Refusal(f'Upstream revision differs: {pin}')
    if command(['git','-C',str(repo),'status','--porcelain','--untracked-files=all']):
        raise Refusal('Upstream checkout has changes or untracked files')
    files={}
    for name in ('run_research_window.py','benchmark_research_lsa.py'):
        path=within(root,root/name)
        if not path.is_file():
            raise Refusal(f'Required kit file missing: {name}')
        files[name]=hash_file(path)
    command(['git','-C',str(repo),'merge-base','--is-ancestor',ORIGINAL_PIN,PIN])
    patch=subprocess.check_output(['git','-C',str(repo),'diff','--binary',ORIGINAL_PIN,PIN])
    return dict(upstream_commit=pin,original_upstream_commit=ORIGINAL_PIN,tree=command(['git','-C',str(repo),'rev-parse','HEAD^{tree}']),research_patch_sha256=hashlib.sha256(patch).hexdigest(),upstream_clean=True,sha256=files)

@contextlib.contextmanager
def kit_lock(root):
    import fcntl
    path=within(root,root/'.window.lock')
    with open(path,'a+',encoding='utf-8') as handle:
        try:
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refusal('Another experiment holds this kit lock') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(),fcntl.LOCK_UN)

def kill_owned_group(proc):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid,signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid,signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5)

def run_child(argv, *, env, cwd, log_path, timeout_seconds):
    with open(log_path,'x',encoding='utf-8') as log:
        proc=subprocess.Popen(argv,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        deadline=time.monotonic()+timeout_seconds
        try:
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    kill_owned_group(proc)
                    return dict(returncode=proc.returncode,timed_out=True,pid=proc.pid)
                try:
                    rc=proc.wait(timeout=min(2,remaining))
                    return dict(returncode=rc,timed_out=False,pid=proc.pid)
                except subprocess.TimeoutExpired:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:
                        kill_owned_group(proc)
                        return dict(returncode=proc.returncode,timed_out=True,pid=proc.pid)
                    try:
                        state=primary_state(timeout=min(5,remaining))
                        if state!='inactive':
                            raise Refusal(f'Primary changed to {state!r} during diagnostic')
                    except Refusal as exc:
                        kill_owned_group(proc)
                        return dict(returncode=proc.returncode,timed_out=False,aborted=True,reason=str(exc),pid=proc.pid)
        except BaseException:
            kill_owned_group(proc)
            raise

def child_command(root,args,output):
    python=root/'.venv/bin/python'
    if not python.is_file():
        raise Refusal('Isolated .venv/bin/python missing')
    return [str(python),str(within(root,root/'benchmark_research_lsa.py')),'--mode','gpu','--allow-gpu','--contexts',*map(str,args.contexts),'--output',str(output),'--max-gpu-mib',str(args.max_gpu_mib),'--max-seconds',str(args.max_seconds-15),'--query-batch',str(args.query_batch),'--repeats',str(args.repeats),'--upstream',str(within(root,root/'upstream-research'))]

def parser():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--plan',action='store_true')
    mode.add_argument('--check-window',action='store_true')
    mode.add_argument('--execute',action='store_true')
    ap.add_argument('--window-label',default='',help='Human coordination reference, not authorization')
    ap.add_argument('--contexts',nargs='+',type=int,default=list(CONTEXTS),choices=CONTEXTS)
    ap.add_argument('--max-gpu-mib',type=int,default=2048)
    ap.add_argument('--max-seconds',type=int,default=900)
    ap.add_argument('--query-batch',type=int,default=64)
    ap.add_argument('--repeats',type=int,default=3)
    return ap

def validate_args(args):
    if not 512 <= args.max_gpu_mib <= 4096:
        raise Refusal('--max-gpu-mib must be512..4096')
    if not 60 <= args.max_seconds <= 1800:
        raise Refusal('--max-seconds must be60..1800')
    if not 1 <= args.query_batch <= 128 or not 1 <= args.repeats <= 10:
        raise Refusal('query-batch must be1..128; repeats must be1..10')
    if len(args.contexts) != len(set(args.contexts)):
        raise Refusal('Repeated contexts not permitted')
    if args.execute and not args.window_label.strip():
        raise Refusal('--execute requires a descriptive --window-label')

def valid_result(result):
    return isinstance(result,dict) and result.get('status')=='passed' and result.get('mode')=='gpu' and result.get('gpu_executed') is True

def execute(root,args):
    if platform.system() != 'Linux':
        raise Refusal('Execution supported only on prepared Linux host')
    with kit_lock(root):
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-')+uuid.uuid4().hex[:8]
        run_dir=within(root,root/'results'/run_id)
        run_dir.mkdir(parents=True,exist_ok=False)
        receipt=dict(schema_version=1,run_id=run_id,started_at=utc(),window_label=args.window_label,status='preflight',gpu_work_started=False,qualification_claim=False,parameters=vars(args))
        code=1
        try:
            receipt['source']=source_provenance(root)
            prep=within(root,root/'PREP-STATUS.json')
            if prep.is_file():
                receipt['prep_status']=json.loads(prep.read_text(encoding='utf-8'))
                receipt['prep_status_sha256']=hash_file(prep)
            receipt['before']=check_window(args.max_gpu_mib)
            output=within(root,run_dir/'benchmark.json')
            argv=child_command(root,args,output)
            receipt['argv']=argv
            cache=within(root,root/'cache')
            cache.mkdir(exist_ok=True)
            env=dict(os.environ)
            env.pop("NVIDIA_VISIBLE_DEVICES",None)
            for key,subdir in [('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TORCH_EXTENSIONS_DIR','torch-extensions'),('TORCHINDUCTOR_CACHE_DIR','torchinductor'),('XDG_CACHE_HOME','xdg')]:
                path=within(root,cache/subdir)
                path.mkdir(exist_ok=True)
                env[key]=str(path)
            env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
            if primary_state() != 'inactive':
                raise Refusal('Primary became active before worker launch')
            receipt['gpu_work_started']=True
            receipt['worker']=run_child(argv,env=env,cwd=root,log_path=run_dir/'worker.log',timeout_seconds=args.max_seconds)
            if receipt['worker']['timed_out']:
                receipt['status']='timeout'
            elif receipt['worker'].get('aborted'):
                receipt['status']='aborted'
                receipt['error']=receipt['worker']['reason']
            elif receipt['worker']['returncode'] != 0:
                receipt['status']='failed'
            elif not output.is_file():
                receipt.update(status='failed',error='Worker returned zero without result')
            else:
                result=json.loads(output.read_text(encoding='utf-8'))
                receipt['benchmark_sha256']=hash_file(output)
                receipt['benchmark_status']=result.get('status') if isinstance(result,dict) else None
                if not valid_result(result):
                    raise ValueError('Worker result does not establish a passed GPU diagnostic')
                receipt['status']='diagnostic_completed'
                code=0
        except (Refusal,OSError,ValueError) as exc:
            receipt.update(status='refused' if not receipt['gpu_work_started'] else 'failed',error=str(exc))
        except BaseException as exc:
            receipt.update(status='interrupted',error=type(exc).__name__)
            raise
        finally:
            with contextlib.suppress(Exception):
                receipt['after']=dict(gpu=gpu_snapshot(),primary=primary_state())
            receipt['finished_at']=utc()
            (run_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(dict(status=receipt['status'],receipt=str(run_dir/'receipt.json'))))
        return code

def main(argv=None):
    args=parser().parse_args(argv)
    root=Path(__file__).resolve().parent
    try:
        validate_args(args)
        if args.execute:
            def terminate_owned(signum, frame):
                raise KeyboardInterrupt(f'Signal {signum}')
            old_handler=signal.signal(signal.SIGTERM,terminate_owned)
            try:
                return execute(root,args)
            finally:
                signal.signal(signal.SIGTERM,old_handler)
        if args.check_window:
            if platform.system() != 'Linux':
                raise Refusal('--check-window requires prepared Linux host')
            print(json.dumps(dict(source=source_provenance(root),samples=check_window(args.max_gpu_mib)),indent=2))
            return 0
        print(json.dumps(dict(mode='plan',gpu_work_started=False,upstream_commit=PIN,contexts=args.contexts,max_gpu_mib=args.max_gpu_mib,hard_timeout_seconds=args.max_seconds,query_batch=args.query_batch,repeats=args.repeats,requirements=['coordinated maintenance/recovery window',f'{PRIMARY} already inactive','compute12.0','three samples <=5% GPU utilization','at least8192MiB free','isolated Python environment'],qualification_claim=False,next='--check-window reads conditions; --execute also requires descriptive --window-label'),indent=2))
        return 0
    except Refusal as exc:
        print(json.dumps(dict(status='refused',error=str(exc))),file=sys.stderr)
        return 2
if __name__=='__main__':
    raise SystemExit(main())
