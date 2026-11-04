#!/usr/bin/env python3
"""Normal authenticated production checks; never changes services or readiness."""
import argparse
import hashlib
import importlib.util
import json
import signal
import sys
import urllib.request
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/USER/Documents/strata-0156-tuning-20261103/promotion-20261103/production_canary.py')
spec=importlib.util.spec_from_file_location('retained_production_canary_helpers',OLD)
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
API=P.API;B=API.CLI


def capacity(h,m,s):
 for k,v in {'status':'ok','service':'strata','loaded':True,'max_context':262144,'images':True,'api_key':True,'qualified':True}.items():
  if h.get(k)!=v:raise B.Refusal('production health mismatch: '+k)
 for k,v in {'engine':'0.1.56','context':262144,'batch_slots':8,'kv':'int8','kv_resident':0,'expert_slots':24576,
             'prefill_cap':8192,'spec':8,'mtp_max':6,'lookup':3,'cvec':'project:4-44'}.items():
  if m.get('engine',{}).get(k)!=v:raise B.Refusal('production capacity mismatch: '+k)
 if s.get('context',{}).get('native')!=262144 or s.get('concurrency',{}).get('serving')!=8 or not s.get('vision',{}).get('available'):
  raise B.Refusal('production advertised context/vision differs')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if not a.execute:print(json.dumps({'execute':False,'context':262144,'slots':8,'normal_gate':True}));return 0
 urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
 cmod=P.helper();c=cmod.load_config();server=json.loads(Path(c['server_config']).read_text())
 if c['context_length']!=262144 or c['max_streams']!=8 or c['environment'].get('ONYX_STRATA_READINESS_GATE')!='1':raise B.Refusal('new normal production contract is required')
 for flag,value in (('--max-context','262144'),('--kv-resident','262144'),('--prefill','8192')):
  if B.arg_value(server['args'],flag)!=value:raise B.Refusal('production argument mismatch: '+flag)
 if B.sha(server['exe'])!='21cb8dd0c744c84176a019ef3cfebd27997968f211b197429df897cc6af92788':raise B.Refusal('official engine identity differs')
 cmod.verify_static(c);guard=P.Guard(cmod,c,server,600);guard.check(force=True)
 directory=a.output_dir.resolve()
 if not directory.is_relative_to(ROOT):raise B.Refusal('evidence directory must remain in owned project')
 directory.mkdir(mode=0o700,exist_ok=False)
 r={'status':'running','started_at':B.utc(),'normal_gate_preserved':True,'services_changed':False,
    'context':262144,'slots':8,'prefill':8192,'initial_probe':guard.snapshot,
    'configuration_sha256':guard.server_sha,'contract_sha256':guard.config_sha,
    'runner_source_sha256':B.sha(Path(__file__)),'cases':[],
    'scope':'Production features and near-capacity text; sustained eight-deep generation is not qualified or retested.'}
 def req(path,payload=None,timeout=120):return P.request(cmod,c,guard,path,payload,timeout)
 try:
  h=req('/health')['response'];m=req('/metrics')['response'];s=req('/v1/status')['response'];capacity(h,m,s)
  r.update(initial_health=h,initial_metrics=m)
  reject=API.http_json(c['base_url'],'/v1/status',None,guard,timeout=10,expected_status=401)
  r['cases'].append({'id':'unauthenticated_status_rejected','status':'passed',**reject})
  image=API.make_image();(directory/'fresh-square.png').write_bytes(image)
  r['image_sha256']=hashlib.sha256(image).hexdigest()
  for name,payload in API.request_cases(c['model'],image):
   result=req('/v1/chat/completions',payload,180);row={'id':name,'status':'failed',**result};r['cases'].append(row)
   API.validate_completion(name,result['response']);row['status']='passed'
   print(json.dumps({'case':name,'status':'passed'}),flush=True)
  case=json.loads((ROOT/'api-long-case.json').read_text())
  if case['prompt_tokens']!=259999 or len(case['tokens'])!=259999:raise B.Refusal('frozen near-capacity fixture differs')
  payload={'model':c['model'],'messages':case['messages'],'max_tokens':case['max_new'],'temperature':0,'stream':False,
           'reasoning_effort':'none','experimental_speed_projection':True,'chat_template_kwargs':{'enable_thinking':False}}
  for name in ('production_near_capacity_259999','production_repeat_near_capacity'):
   result=req('/v1/chat/completions',payload,300);value=result['response'];choice=value['choices'][0]
   row={'id':name,'status':'failed','token_sha256':case['token_sha256'],'expected':case['expected'],**result};r['cases'].append(row)
   if value.get('usage',{}).get('prompt_tokens')!=259999 or choice.get('finish_reason')!='stop' or not B.check_answer(case,(choice['message'].get('content') or '').strip()):
    raise B.Refusal('production full-token-count retrieval/finish differs')
   row['status']='passed';print(json.dumps({'case':name,'status':'passed','usage':value.get('usage')}),flush=True)
  r['repeat_cached_tokens']=r['cases'][-1]['response'].get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens')
  guard.check(force=True);cmod.verify_static(c)
  h=req('/health')['response'];m=req('/metrics')['response'];s=req('/v1/status')['response'];capacity(h,m,s)
  r.update(final_health=h,final_metrics=m,final_probe=guard.snapshot,identity_preserved=True,status='passed')
 except BaseException as error:r.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=type(error).__name__+': '+str(error))
 finally:r['finished_at']=B.utc();API.private_json(directory/'report.json',r)
 print(json.dumps({'status':r['status'],'report':str(directory/'report.json')}),flush=True);return 0 if r['status']=='passed' else 1


if __name__=='__main__':
 def interrupt(*_):raise KeyboardInterrupt('production canary interrupted')
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGHUP,interrupt)
 raise SystemExit(main())
