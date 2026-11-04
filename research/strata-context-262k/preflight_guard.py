#!/usr/bin/env python3
"""Read-only guard for isolated capacity tests; never changes production policy."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
OWNER='<maintenance-owner>'
PRODUCTION='qwen38-flash-next-strata.service'
RECOVERY=Path('/var/lib/onyx-inference-recovery')
BASELINE=ROOT/'initial-state.json'
COMPETITORS=('qwen38-flash-next-w4-sidecar.service','qwen38-flash-next-tensorfold.service','vllm.service',
             'onyx-personal-session.service','onyx-personal-proxy.service')

class Refusal(RuntimeError):pass

def command(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=20)
 if p.returncode:raise Refusal('required read failed: '+Path(argv[0]).name)
 return p.stdout.strip()

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def process(pid):
 try:
  p=Path('/proc')/str(pid); fields=(p/'stat').read_text().rsplit(')',1)[1].split()
  status=dict(line.split(':',1) for line in (p/'status').read_text().splitlines() if ':' in line)
  return {'pid':pid,'start_ticks':fields[19],'ppid':int(fields[1]),'pgid':int(fields[2]),'sid':int(fields[3]),
          'uid':int(status['Uid'].split()[1]),'exe':str((p/'exe').resolve(strict=True))}
 except (FileNotFoundError,ProcessLookupError):return None

def unit(name,user=None):
 argv=['systemctl','show',name,'--property=ActiveState','--property=SubState','--property=MainPID','--property=UnitFileState']
 if user:
  uid=pwd.getpwnam(user).pw_uid; runtime='/run/user/'+str(uid)
  argv=['runuser','-u',user,'--','env','XDG_RUNTIME_DIR='+runtime,'DBUS_SESSION_BUS_ADDRESS=unix:path='+runtime+'/bus',
        'systemctl','--user',*argv[1:]]
 return dict(line.split('=',1) for line in command(argv).splitlines() if '=' in line)

def snapshot(pids=()):
 """Protected file/status reads only. No generation or mutation commands."""
 if os.geteuid()!=0:raise Refusal('snapshot requires root read access')
 baseline=json.loads(BASELINE.read_text())
 files=baseline['production_config_sha256']
 state=json.loads((RECOVERY/'state.json').read_text())
 holdpath=RECOVERY/'HOLD'
 hold=None
 if holdpath.exists():
  info=holdpath.lstat()
  if holdpath.is_symlink() or info.st_uid!=0 or info.st_mode&0o077:raise Refusal('unsafe maintenance HOLD')
  hold=json.loads(holdpath.read_text())
 contract=json.loads(Path('/etc/onyx-inference/strata.json').read_text())
 gpu=[]
 for line in command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).splitlines():
  if not line.strip().isdigit():raise Refusal('unrecognized GPU ownership record')
  pid=int(line); identity=process(pid)
  if identity is None:raise Refusal('GPU process changed during inspection')
  gpu.append(identity)
 memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
 allowed=[baseline['production_profile']['exe'],baseline['production_profile']['vision_exe'],contract['exec_start'][0]]
 resolved={str(Path(p).resolve()):sha(p) for p in allowed}
 for filename in allowed[:2]:
  if contract['hashes'].get(filename)!=resolved[str(Path(filename).resolve())]:
   raise Refusal('official private executable differs from the pinned production bundle')
 return {'hashes':{p:sha(p) for p in files},'hold':hold,'production':unit(PRODUCTION),
  'competitors':{name:unit(name) for name in COMPETITORS},'sunshine':unit('sunshine.service',contract['user']),
  'attempts':state['attempts'],'capacity_block':state.get('capacity_block'),
  'production_listener':command(['ss','-H','-ltnp','sport = :8002']),
  'gpu':gpu,'requested_processes':{str(pid):process(pid) for pid in pids},
  'allowed_executables':resolved,'gpu_free_mib':int(command(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'])),
  'host_available_kib':int(memory['MemAvailable'].split()[0])}

class Guard:
 """NativeEngine-compatible check interface, with explicit private Popen binding.

 Call bind_engine(engine_or_wrapper) BEFORE its context manager, or call
 register_process(proc) immediately after a direct start_new_session Popen.
 Owned processes must be direct children and create their own session. GPU
 children must belong to that exact live session and a pinned official binary.
 """
 def __init__(self,owner=OWNER,config_sha=None,contract_sha=None,total_seconds=3600,*,baseline_path=BASELINE,snapshot_reader=None):
  if owner!=OWNER:raise Refusal('unexpected capacity-test maintenance owner')
  if not math.isfinite(total_seconds) or not 0<total_seconds<=21600:raise Refusal('invalid bounded guard deadline')
  self.owner=owner; self.baseline=json.loads(Path(baseline_path).read_text())
  hashes=self.baseline['production_config_sha256']
  if config_sha and hashes['/etc/onyx-inference/strata-server.json']!=config_sha:raise Refusal('server hash differs from captured baseline')
  if contract_sha and hashes['/etc/onyx-inference/strata.json']!=contract_sha:raise Refusal('contract hash differs from captured baseline')
  self.deadline=time.monotonic()+total_seconds; self.last_check=0; self.anchor=None; self.provider=None
  self.reader=snapshot_reader or self._snapshot; self.caller_pid=os.getpid(); self.caller_uid=os.geteuid()
 def _snapshot(self,pids):
  if os.geteuid()==0:return snapshot(pids)
  argv=['sudo','-n','/usr/bin/python3','-B',str(Path(__file__).resolve()),'_snapshot']
  for pid in pids:argv+=['--pid',str(pid)]
  return json.loads(command(argv))
 def bind_engine(self,engine):
  self.provider=lambda:getattr(engine,'proc',None)
  self.anchor=None
 def register_process(self,proc):
  if proc is None or proc.poll() is not None:raise Refusal('private child has already exited')
  view=self.reader([proc.pid]); self._claim(proc.pid,view); self.provider=lambda:proc
 def _claim(self,pid,view):
  identity=view['requested_processes'].get(str(pid))
  if not identity or identity['ppid']!=self.caller_pid or identity['pgid']!=pid or identity['sid']!=pid:
   raise Refusal('private process is not our direct child in its own session')
  if identity['uid']!=self.caller_uid or identity['exe'] not in view['allowed_executables']:
   raise Refusal('private process executable/account differs from pinned baseline')
  self.anchor=identity
 def check(self,force=False,before_start=False):
  now=time.monotonic()
  if now>=self.deadline:raise Refusal('capacity-test deadline expired')
  if not force and not before_start and now-self.last_check<2:return
  self.last_check=now
  proc=self.provider() if self.provider else None
  pids=[proc.pid] if proc is not None else ([self.anchor['pid']] if self.anchor else [])
  view=self.reader(pids)
  if view['hashes']!=self.baseline['production_config_sha256']:raise Refusal('production/personal/shared config changed')
  hold=view['hold'] or {}
  if hold.get('kind')!='owned-strata-maintenance' or hold.get('version')!=1 or hold.get('owner')!=self.owner:
   raise Refusal('exact owned capacity-test HOLD is absent')
  primary=view['production']
  if primary.get('ActiveState')!='inactive' or primary.get('MainPID')!='0' or view['production_listener']:
   raise Refusal('production is not fully inactive without its listener')
  if any(u.get('ActiveState') not in ('inactive','failed') or u.get('MainPID')!='0' for u in view['competitors'].values()):
   raise Refusal('a competing inference/personal service is active')
  sun=view['sunshine']
  if sun.get('ActiveState')!='inactive' or sun.get('MainPID')!='0' or sun.get('UnitFileState')!=self.baseline['sunshine_user_unit'].get('UnitFileState'):
   raise Refusal('Sunshine changed; keep it stopped and preserve enablement')
  if view['attempts']!=self.baseline['recovery']['attempts'] or view['capacity_block']:
   raise Refusal('recovery attempt history/capacity latch changed during private work')
  if view['host_available_kib']<4*1024*1024:raise Refusal('host available memory below 4 GiB')
  if before_start:
   if view['gpu']:raise Refusal('another compute process owns the GPU before private start')
   if view['gpu_free_mib']<80000:raise Refusal('less than 80000 MiB free before isolated full-model start')
   return
  if proc is not None:
   current=view['requested_processes'].get(str(proc.pid))
   if self.anchor is None and proc.poll() is None:self._claim(proc.pid,view)
   elif self.anchor and current!=self.anchor:raise Refusal('private root process identity changed/exited')
  if view['gpu'] and not self.anchor:raise Refusal('GPU use has no explicitly registered private owner')
  for gpu in view['gpu']:
   anchor=view['requested_processes'].get(str(self.anchor['pid'])) if self.anchor else None
   if anchor!=self.anchor or gpu['sid']!=self.anchor['sid'] or gpu['pgid']!=self.anchor['pgid'] or gpu['uid']!=self.anchor['uid']:
    raise Refusal('GPU process belongs to another session/account')
   if gpu['exe'] not in view['allowed_executables']:raise Refusal('GPU process is not a pinned official executable')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('_snapshot','check-before-start'))
 p.add_argument('--pid',type=int,action='append',default=[]);a=p.parse_args()
 try:
  if a.action=='_snapshot':print(json.dumps(snapshot(a.pid)))
  else:Guard().check(force=True,before_start=True);print('CONTEXT262K_PREFLIGHT_OK')
 except (Refusal,OSError,ValueError,subprocess.SubprocessError) as error:
  print(json.dumps({'status':'refused','error':str(error)}),file=sys.stderr);raise SystemExit(2)
