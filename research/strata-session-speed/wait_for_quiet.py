#!/usr/bin/env python3
"""Observe a bounded quiet gap; never pauses or cancels production traffic."""
import datetime as dt,importlib.util,json,os,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('quiet_gap_contract','/usr/local/libexec/onyx-inference-recovery/strata-contract.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C);c=C.load_config();started=time.monotonic();last=-60;quiet=0;rows=[]
while time.monotonic()-started<180:
 m=C.request(c,'/metrics');l=m['live'];row={'at':dt.datetime.now(dt.timezone.utc).isoformat(),'running':l.get('running'),'queued':l.get('queued'),'admission_inflight':l.get('admission_inflight'),'state':l.get('state')};rows.append(row)
 idle=l.get('state')=='idle' and all(l.get(k,0)==0 for k in ['running','queued','waiting','admitting','admission_inflight'])
 quiet=quiet+1 if idle else 0
 if time.monotonic()-last>=30:print(json.dumps(row),flush=True);last=time.monotonic()
 if quiet>=3:
  print(json.dumps({'status':'quiet_gap_observed','at':row['at']}),flush=True);break
 time.sleep(2)
else:
 os.umask(0o077);p=r/'receipts/quiet-gap-deferred.json';p.write_text(json.dumps({'status':'deferred_to_existing_production_traffic','observations':rows,'synthetic_requests_sent':0,'production_changed':False},indent=2)+'\n');print(json.dumps({'status':'deferred_to_existing_production_traffic'}),flush=True);raise SystemExit(0)
python='/home/USER/model-tests/strata-20261005/Strata/.venv/bin/python'
for directory,options in [('live-C2-r1',['--concurrency','2','--rounds','2','--total-seconds','180','--request-seconds','60']),('live-baseline-r3',['--rounds','2','--total-seconds','480','--request-seconds','75'])]:
 p=subprocess.run([python,'-B',str(r/'live_benchmark.py'),'--execute','--output-dir',str(r/'receipts'/directory),*options])
 if p.returncode:break
