#!/usr/bin/env python3
"""Independent post-promotion262 file/GET audit; no generation or service actions."""
import copy,datetime as dt,hashlib,importlib.util,json,os,pwd,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent
EXPECTED={
 '/etc/onyx-inference/strata-server.json':'<private-configuration-sha256>',
 '/etc/onyx-inference/strata.json':'<private-configuration-sha256>',
 '/etc/onyx-personal/config.json':'<private-configuration-sha256>',
 '/usr/local/libexec/onyx-inference-recovery/strata-contract.py':'bd6ea9561ab8d192daea7620f92f5172bf09953ed15de49816297a3a76ac84de',
 '/etc/onyx-inference/strata-server.shared-settings.json':'<private-configuration-sha256>'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text())
def main():
 if os.geteuid()!=0:raise SystemExit('Root read access required; no mutation commands implemented')
 spec=importlib.util.spec_from_file_location('strict_live262_contract','/usr/local/libexec/onyx-inference-recovery/strata-contract.py')
 C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
 c=C.load_config();C.verify_static(c);snap=C.probe(c)
 metrics=C.request(c,'/metrics')['engine'];status=C.request(c,'/v1/status')
 server=load('/etc/onyx-inference/strata-server.json');personal=load('/etc/onyx-personal/config.json')
 initial=load(PROJECT/'initial-state.json');prior=load(PROJECT/'private-fleet/restoration-audit.json')
 state=load('/var/lib/onyx-inference-recovery/state.json');old_attempts=prior['recovery']['attempts_after'];attempts=state['attempts']
 added=[v for v in attempts if v not in old_attempts]
 hashes={p:sha(p) for p in EXPECTED};pins={p:sha(p)==h for p,h in personal['baseline_sha256'].items()}
 desired_args=list(initial['production_profile']['args'])
 for flag,value in [('--max-context','262144'),('--kv-resident','262144'),('--prefill','8192')]:
  if desired_args.count(flag)!=1:raise RuntimeError('Ambiguous baseline flag')
  desired_args[desired_args.index(flag)+1]=value
 q=personal['qualification'];backup=Path(q['primary_promotion']);before_personal=load(backup/'personal-config.json')
 untouched_personal=lambda x:{k:v for k,v in x.items() if k not in ('qualification','baseline_sha256')}
 uid=pwd.getpwnam(c['user']).pw_uid;runtime='/run/user/'+str(uid)
 argv=['runuser','-u',c['user'],'--','env','XDG_RUNTIME_DIR='+runtime,'DBUS_SESSION_BUS_ADDRESS=unix:path='+runtime+'/bus',
  'systemctl','--user','show','sunshine.service','--property=ActiveState','--property=SubState','--property=MainPID','--property=UnitFileState']
 result=subprocess.run(argv,capture_output=True,text=True,timeout=20,check=True)
 sun=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
 result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20,check=True)
 gpu=[]
 for line in result.stdout.splitlines():
  if not line.strip().isdigit():raise RuntimeError('Malformed GPU PID metadata')
  pid=int(line.strip());C._owned_pid(pid,c);gpu.append(pid)
 members=Path('/sys/fs/cgroup/system.slice')/c['unit']/'cgroup.procs'
 vision=[]
 for value in members.read_text().split():
  pid=int(value);p=Path('/proc')/value
  try:
   if (p/'exe').resolve()==Path(server['vision']['exe']).resolve():C._owned_pid(pid,c);vision.append(pid)
  except FileNotFoundError:continue
 private_files=[str(p.relative_to(PROJECT)) for p in PROJECT.rglob('candidate-private.json') if p.exists()]
 limited=load(c['qualification_receipt'])
 checks={
  'five_approved_hashes_match':hashes==EXPECTED,
  'strict_context_contract262144x8':c['context_length']==262144 and c['max_streams']==8,
  'actual_full_resident_int8_capacity':all(metrics.get(k)==v for k,v in {'context':262144,'batch_slots':8,'kv':'int8','kv_resident':0,'expert_slots':24576,'prefill_cap':8192}.items()),
  'retained_native_mtp_control_vector':all(metrics.get(k)==v for k,v in {'engine':'0.1.56','spec':8,'mtp_max':6,'lookup':3,'cvec':'project:4-44'}.items()),
  'wrapper_source_unchanged':c['source_commit']==initial['production_contract']['source_commit'],
  'only_three_requested_argument_changes':server['args']==desired_args,
  'model_profile_environment_unchanged':server['env']==initial['production_profile']['env'] and server['parallel']==8,
  'same_official_engine_vision_libraries':server['exe']==initial['production_profile']['exe'] and server['vision']['exe']==initial['production_profile']['vision_exe'] and server['lib_dirs']==initial['production_profile']['lib_dirs'],
  'all_personal_baseline_pins_match':all(pins.values()),
  'personal_primary_and_helper_pins_updated':q['primary_contract']==hashes['/etc/onyx-inference/strata.json'] and personal['baseline_sha256']['/usr/local/libexec/onyx-inference-recovery/strata-contract.py']==EXPECTED['/usr/local/libexec/onyx-inference-recovery/strata-contract.py'],
  'personal_model_settings_unchanged':untouched_personal(personal)==untouched_personal(before_personal),
  'normal_gate_and_identity_bound_receipt':c['environment'].get('ONYX_STRATA_READINESS_GATE')=='1' and snap['health']['qualified'] is True and snap['ready']['identity']==snap['identity'] and snap['ready']['config_sha256']==hashes['/etc/onyx-inference/strata.json'] and snap['ready']['canary']=='SIDECAR_OK',
  'advertised_context_and_vision':status.get('context',{}).get('native')==262144 and status.get('concurrency',{}).get('serving')==8 and status.get('vision',{}).get('available') is True and len(vision)==1,
  'old_attempts_retained_plus_one_legal_start':all(t in attempts for t in old_attempts) and len(added)==1 and added[0]-max(old_attempts)>=900,
  'healthy_recovery_without_hold_or_capacity_latch':state.get('status')=='healthy' and state.get('failures')==0 and not state.get('capacity_block') and not Path('/var/lib/onyx-inference-recovery/HOLD').exists(),
  'private_credentials_removed_gpu_only_managed':not private_files and bool(gpu),
  'negative_eight_deep_qualification_preserved':limited.get('eight_deep_simultaneous_generation_qualified') is False and c.get('active_tuning_profile',{}).get('eight_deep_simultaneous_generation_qualified') is False,
  'no_lsa':not any(k.startswith('STRATA_LSA') for k in {**c['environment'],**server['env']}),
  'sunshine_stays_off_enabled_unchanged':sun.get('ActiveState')=='inactive' and sun.get('SubState')=='dead' and sun.get('MainPID')=='0' and sun.get('UnitFileState')==initial['sunshine_user_unit']['UnitFileState'],
 }
 report={'kind':'independent read-only live262 promotion audit','at':dt.datetime.now(dt.timezone.utc).isoformat(),
  'status':'passed' if all(checks.values()) else 'failed','checks':checks,'configuration_sha256':hashes,
  'personal_pin_results':pins,'managed_identity':snap['identity'],'vision_pid':vision,'managed_gpu_compute_pids':gpu,
  'service':{k:snap['unit'][k] for k in ('ActiveState','SubState','MainPID','UnitFileState','Restart')},
  'capacity':{k:metrics.get(k) for k in ('context','batch_slots','kv','kv_resident','expert_slots','prefill_cap','engine','spec','mtp_max','lookup','cvec')},
  'readiness':{'qualified':snap['health']['qualified'],'canary':snap['ready']['canary'],'config_sha256':snap['ready']['config_sha256'],'verified_epoch':snap['ready']['verified_epoch']},
  'recovery':{'status':state.get('status'),'attempts_before':old_attempts,'attempts_after':attempts,'new_managed_start':added,'capacity_block':state.get('capacity_block')},
  'sunshine_user_unit':sun,'candidate_private_files_remaining':private_files,
  'scope_limit':'Eight-deep sustained generation remains unqualified; operator promotion does not change that evidence.',
  'completion_requests_sent':0,'service_or_configuration_actions_performed':0}
 destination=ROOT/'independent-live-audit.json'
 with destination.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':report['status'],'checks':checks,'path':str(destination)},indent=2))
 if not all(checks.values()):raise SystemExit(2)
if __name__=='__main__':main()
