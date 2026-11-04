#!/usr/bin/env python3
"""Apply the explicitly approved262K/C8 memory-fit profile under owned maintenance."""
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
sys.dont_write_bytecode=True

ROOT=Path('/home/USER/Documents/strata-context-262k-20261103')
OWNER='<maintenance-owner>'
SERVER=Path('/etc/onyx-inference/strata-server.json')
CONTRACT=Path('/etc/onyx-inference/strata.json')
PERSONAL=Path('/etc/onyx-personal/config.json')
HELPER=Path('/usr/local/libexec/onyx-inference-recovery/strata-contract.py')
SHARED=Path('/etc/onyx-inference/strata-server.shared-settings.json')
EXPECTED={SERVER:'<private-configuration-sha256>',
 CONTRACT:'<private-configuration-sha256>',
 PERSONAL:'<private-configuration-sha256>',
 HELPER:'4420f5dfd1fc1d039dacade988851a279cd7f00858e09ff2effdbdf0798ab540',
 SHARED:'<private-configuration-sha256>'}
NEW_HELPER_SHA='bd6ea9561ab8d192daea7620f92f5172bf09953ed15de49816297a3a76ac84de'


def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def encoded(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()


def replace(args,flag,value):
 result=list(args);assert result.count(flag)==1,'ambiguous/missing argument: '+flag
 result[result.index(flag)+1]=str(value);return result


def main():
 assert os.geteuid()==0,'root required';os.umask(0o077)
 old_tools=Path('/home/USER/Documents/strata-0156-tuning-20261103/promotion-20261103/deploy.py')
 assert sha(old_tools)=='cc02bdeb5316625714e030e2151d9798b84ddd4235acf698797ffc32536c91df'
 D=module('verified_atomic_deployment_helpers',old_tools)
 maintenance=module('context_promotion_maintenance',HELPER.with_name('strata-maintenance.py'));ctl=maintenance.load_controller()
 at=dt.datetime.now(dt.timezone.utc);backup=Path('/var/lib/onyx-inference-recovery/backups')/('strata-context262k-promote-'+at.strftime('%Y%m%dT%H%M%SZ'))
 stamp=at.isoformat()
 with contextlib.ExitStack() as stack:
  for filename in ctl.LOCKS:
   lock=stack.enter_context(open(filename,'a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  hold=maintenance.require_owner(OWNER);old=ctl.contract.load_config();ctl.contract.verify_static(old)
  maintenance.inactive_personal(ctl);assert maintenance.fully_stopped(ctl,old)
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU is owned'
  for p,h in EXPECTED.items():assert sha(p)==h,'concurrent production edit: '+str(p)
  draft=ROOT/'promotion262/strata-contract.py';helper_bytes=draft.read_bytes();assert hashlib.sha256(helper_bytes).hexdigest()==NEW_HELPER_SHA
  native=json.loads((ROOT/'native-r2/report.json').read_text());arm=native['arms'][0]
  assert native['status']=='passed' and arm['slots']==8 and arm['prefill']==8192 and arm['context']==262144
  assert arm['infrastructure_status']=='passed' and arm['quality_status']=='passed' and arm['owned_process_group_empty']
  api=json.loads((ROOT/'api8-r2/report.json').read_text())
  assert len(api['cases'])==6 and all(x['status']=='passed' for x in api['cases'])
  assert api['initial_metrics']['engine']['expert_slots']==24576 and api['initial_health']['images'] is True
  assert api['initial_metrics']['engine']['context']==262144 and api['initial_metrics']['engine']['prefill_cap']==8192
  assert api['owned_process_group_empty'] and api['private_credential_file_removed']
  for n in ('api8-r1','api8-r2','api8-r3','api8-r4'):
   report=json.loads((ROOT/n/'report.json').read_text());assert report['owned_process_group_empty'] and report['private_credential_file_removed']
  assert not list(ROOT.glob('api*/candidate-private.json'))
  profile_bytes=(ROOT/'memory-fit-profile.json').read_bytes();profile=json.loads(profile_bytes)
  assert profile['parallel']==8 and {k:str(v) for k,v in profile['arg_overrides'].items()}=={'--max-context':'262144','--kv-resident':'262144','--prefill':'8192'} and profile['env']=={}
  backup.mkdir(mode=0o700)
  originals={p:p.read_bytes() for p in EXPECTED};modes={p:stat.S_IMODE(p.stat().st_mode) for p in EXPECTED}
  for p,data in originals.items():D.durable(backup/('personal-config.json' if p==PERSONAL else p.name),data)
  for p in (Path('/var/lib/onyx-inference-recovery/HOLD'),Path('/var/lib/onyx-inference-recovery/state.json')):
   D.durable(backup/('before-'+p.name),p.read_bytes())
  D.durable(backup/'selected-profile.json',profile_bytes)
  proof_names=['native-r1/report.json','native-r2/report.json','api8-r1/report.json','api8-r2/report.json',
               'api8-r3/report.json','api8-r4/report.json','promotion262/cpu-tests.log',
               'promotion262/strata-contract-context.patch','promotion262/test_context_contract.py']
  evidence={}
  for i,name in enumerate(proof_names):
   data=(ROOT/name).read_bytes();digest=hashlib.sha256(data).hexdigest();evidence[name]=digest
   D.durable(backup/('evidence-%02d-'%i+Path(name).name),data)
  receipt=backup/'operator-qualification.json'
  qualification={'version':1,'at':stamp,'authorization':'Jeff: No make this the new production.',
   'status':'operator-approved limited qualification; managed startup and production canaries required',
   'context':262144,'slots':8,'prefill':8192,'engine':'0.1.56','model_weights_changed':False,
   'native_retrieval_and_reuse':'passed','normal_private_API_features':'6/6 passed including259999-token exact retrieval and matched vision',
   'eight_deep_simultaneous_generation_qualified':False,'maximum_observed_deep_decoding':4,
   'stress_limit':'Cold overlap run ended on natural output completion; one-token priming run did not establish eight retained slot images and reached its deadline. No memory-failure conclusion.',
   'rejected_profile':'8slots/262144/prefill16384 with vision demoted expert cache to24007; selected8192 retains24576',
   'experimental_rope_scaling':False,'LSA_enabled':False,'evidence_sha256':evidence,
   'selected_profile_sha256':hashlib.sha256(profile_bytes).hexdigest(),'previous_contract_sha256':EXPECTED[CONTRACT],
   'Sunshine':'keep stopped; enablement unchanged'}
  D.durable(receipt,encoded(qualification))
  server=json.loads(originals[SERVER]);contract=json.loads(originals[CONTRACT]);personal=json.loads(originals[PERSONAL])
  assert server['parallel']==8 and contract['context_length']==131072 and contract['max_streams']==8
  assert contract['engine_argv']==[server['exe'],'--serve',*server['args'],'--batch','8']
  old_args=list(server['args'])
  for flag,value in profile['arg_overrides'].items():server['args']=replace(server['args'],flag,value)
  expected_native=[x for x in native['schedule'][0]['args']]
  assert expected_native==[*server['args'],'--batch','8'],'selected production arguments differ from native evidence'
  server_bytes=encoded(server);server_sha=hashlib.sha256(server_bytes).hexdigest()
  contract['context_length']=262144;contract['engine_argv']=[server['exe'],'--serve',*server['args'],'--batch','8']
  contract['hashes'][str(SERVER)]=server_sha;contract['hashes'][str(HELPER)]=NEW_HELPER_SHA
  contract['hashes'][str(receipt)]=sha(receipt);contract['hashes'][str(backup/'selected-profile.json')]=sha(backup/'selected-profile.json')
  contract['approved_by']='Jeff explicitly requested262K production after the reported test limitations: No make this the new production. '+stamp
  contract['qualification_receipt']=str(receipt)
  contract['runtime_version']='Retained Python0.1.42 compatibility runtime; signed official engine/vision0.1.56 CUDA13.0; eight262144-token sessions; prefill8192; operator-approved limited context qualification'
  contract.setdefault('tuning_history',[]).append({'at':stamp,'previous_profile':contract.get('active_tuning_profile'),
      'reason':'Explicit operator promotion of tested262K/C8 memory-fit profile','previous_contract_sha256':EXPECTED[CONTRACT]})
  contract['active_tuning_profile']={'name':'fused_prefill_262k_c8_memory_fit','selected_at':stamp,
    'context':262144,'slots':8,'prefill_chunk':8192,'engine_environment_delta':{},'selected_profile':str(backup/'selected-profile.json'),
    'profile_sha256':sha(backup/'selected-profile.json'),'qualification_receipt':str(receipt),
    'eight_deep_simultaneous_generation_qualified':False,'status':'explicit operator-approved production profile; limited research scope; live readiness separate'}
  contract.setdefault('reseal_history',[]).append({'at':stamp,'backup':str(backup),'reason':'operator-approved262K context default',
     'previous_contract_sha256':EXPECTED[CONTRACT],'previous_server_sha256':EXPECTED[SERVER],
     'capacity_guard_old_sha256':EXPECTED[HELPER],'capacity_guard_new_sha256':NEW_HELPER_SHA,'wrapper_unchanged':True})
  contract_bytes=encoded(contract);contract_sha=hashlib.sha256(contract_bytes).hexdigest()
  assert str(PERSONAL) not in contract['hashes']
  assert personal['baseline_sha256'][str(CONTRACT)]==EXPECTED[CONTRACT] and personal['baseline_sha256'][str(HELPER)]==EXPECTED[HELPER]
  personal['baseline_sha256'][str(CONTRACT)]=contract_sha;personal['baseline_sha256'][str(HELPER)]=NEW_HELPER_SHA
  if str(SERVER) in personal['baseline_sha256']:personal['baseline_sha256'][str(SERVER)]=server_sha
  q=personal.setdefault('qualification',{});q.update(primary_contract=contract_sha,primary_promotion=str(backup),
    primary_runtime='Official engine/vision0.1.56; retained wrapper0.1.42; eight262144-token sessions; prefill8192; spec6')
  q['context_update_262k']={'at':stamp,'backup':str(backup),'authorization':'Jeff: No make this the new production.',
    'personal_model_changed':False,'gpu_swap_tested':False,'eight_deep_simultaneous_generation_qualified':False}
  new={HELPER:helper_bytes,SERVER:server_bytes,CONTRACT:contract_bytes,PERSONAL:encoded(personal)}
  assert maintenance.require_owner(OWNER)==hold and maintenance.fully_stopped(ctl,old)
  for p,h in EXPECTED.items():assert sha(p)==h,'concurrent edit before transaction: '+str(p)
  attempted=[]
  try:
   for p,data in new.items():attempted.append(p);D.durable(p,data,modes[p])
   fresh=module('fresh_context262_contract',HELPER);ctl.contract=fresh;c=fresh.load_config();fresh.verify_static(c)
   assert c['context_length']==262144 and c['max_streams']==8
   for p,h in personal['baseline_sha256'].items():assert sha(p)==h,'personal pin mismatch: '+p
   assert maintenance.require_owner(OWNER)==hold and maintenance.fully_stopped(ctl,c)
  except BaseException:
   for p in reversed(attempted):
    current=p.read_bytes()
    if current==new[p]:D.durable(p,originals[p],modes[p])
    elif current!=originals[p]:raise RuntimeError('Concurrent bytes preserved during rollback; owned HOLD retained: '+str(p))
   previous=module('restored_context131_contract',HELPER);previous.verify_static(previous.load_config());raise
  result={'version':1,'at':stamp,'status':'deployed_static_verified_owned_HOLD_retained','backup':str(backup),
    'context':262144,'slots':8,'prefill':8192,'qualification_receipt':str(receipt),'rollback_server_and_helper_saved':True,
    'old_sha256':{str(p):h for p,h in EXPECTED.items()},'new_sha256':{str(p):sha(p) for p in new},
    'state_history_modified':False,'unit_changed':False,'runtime_wrapper_changed':False,'Sunshine_modified':False,
    'eight_deep_simultaneous_generation_qualified':False}
  D.durable(backup/'deployment-result.json',encoded(result));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
