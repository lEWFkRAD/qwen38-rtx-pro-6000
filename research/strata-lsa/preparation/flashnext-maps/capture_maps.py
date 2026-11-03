import argparse,hashlib,json,os,signal,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent))
import run_research_window as guard
p=argparse.ArgumentParser();p.add_argument('--documents',nargs='+',required=True);a=p.parse_args()
plan=json.loads((root/'capture-plan.json').read_text());manifest=json.loads((root/'corpus/manifest.json').read_text())
docs={d['id']:d for d in manifest['documents']}
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(plan['engine'])==plan['engine_sha256'],'capture engine changed'
assert sha('/etc/onyx-inference/strata-server.json')==plan['source_config_sha256'],'production config changed'
for asset in plan['model_assets']:
    st=Path(asset['path']).stat();assert st.st_size==asset['size'] and st.st_mtime_ns==asset['mtime_ns'],'model asset changed'
def interrupted(signum,frame):raise KeyboardInterrupt('capture interrupted')
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGHUP,interrupted)
with guard.kit_lock(root.parent):
    guard.check_window(4096)
    for name in a.documents:
        doc=docs[name];assert sha(doc['token_path'])==doc['token_sha256']
        output=root/'captures'/name;output.mkdir(exist_ok=False)
        env={k:v for k,v in os.environ.items() if not k.startswith(('STRATA_','CUDA_','NVIDIA_','LD_'))}
        env.update(plan['env']);env['STRATA_LSA_CAPTURE_DIR']=str(output)
        argv=[plan['engine'],*plan['args'],'--tokens-file',doc['token_path']]
        receipt={'document':doc,'started_at':guard.utc(),'argv':argv,'engine_sha256':plan['engine_sha256'],'production_config_sha256':plan['source_config_sha256']}
        print(json.dumps({'event':'capture_start','document':name,'tokens':doc['tokens']}),flush=True)
        try:
            receipt['worker']=guard.run_child(argv,env=env,cwd=root,log_path=output/'engine.log',timeout_seconds=900)
            files=sorted(output.glob('*.lsacap'))
            receipt['files']=[{'name':f.name,'bytes':f.stat().st_size,'sha256':sha(f)} for f in files]
            expected={f'layer-{l}-p0.lsacap' for l in range(3,48,4)}
            receipt['status']='captured' if receipt['worker']['returncode']==0 and {f.name for f in files}==expected and not list(output.glob('*.partial')) else 'failed'
        finally:
            receipt['finished_at']=guard.utc();(output/'receipt.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps({'event':'capture_finished','document':name,'status':receipt['status'],'files':len(receipt['files']),'worker':receipt['worker']}),flush=True)
        if receipt['status']!='captured':
            print((output/'engine.log').read_text(errors='replace')[-8000:],flush=True)
            raise SystemExit(1)
