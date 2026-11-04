#!/usr/bin/env python3
"""Plan-first local sanitized staging. Never edits Git, uploads, or probes production."""
from __future__ import annotations
import argparse,ast,datetime as dt,hashlib,importlib.util,json,os,re,sys,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=Path('research/strata-session-speed')
REPO=ROOT.parent/'lsa-rtx-publication-20261103'
PIN_ROLES={'/etc/onyx-inference/strata-server.json':'approved-server-config-sha256',
 '/etc/onyx-inference/strata.json':'approved-serving-contract-sha256',
 '/etc/onyx-personal/config.json':'approved-personal-baseline-sha256',
 '/usr/local/libexec/onyx-inference-recovery/strata-contract.py':'approved-contract-helper-sha256',
 '/etc/onyx-inference/strata-server.shared-settings.json':'approved-shared-settings-sha256'}
DROP={'initial_probe','final_probe','approved_configuration_sha256','configuration_sha256','contract_sha256',
 'server_config_sha256','personal_pins','baseline_sha256','process_identity','process_identities','identity','vision_identity',
 'pid','ppid','pgid','sid','start_ticks','response_id','request_id','session_id','actual_content','reasoning_content',
 'initial_health','final_health','initial_status','final_status','initial_metrics','final_metrics','response',
 'api_key','api_keys','compatibility_api_keys','authorization','password','access_token','refresh_token','client_secret'}
RUNTIME=re.compile(r'(?<![A-Za-z0-9])/(?:home|opt|etc|proc|sys|var|usr/local)/[^\s\"\'<>),;]+')
OWNER=re.compile(r'\bcodex-[A-Za-z0-9_-]*\d{8}-[A-Za-z0-9._:-]+\b')
LINK=re.compile(r'(?<!!)\[[^\]\n]+\]\(([^)]+)\)')
class Refusal(RuntimeError):pass

def sha(raw):return hashlib.sha256(raw).hexdigest()
def strict_json(raw):
 def bad(_):raise Refusal('nonfinite JSON')
 def pairs(items):
  out={}
  for k,v in items:
   if k in out:raise Refusal('duplicate JSON key')
   out[k]=v
  return out
 return json.loads(raw,parse_constant=bad,object_pairs_hook=pairs)
def normalizer(path=None):
 if path is None:
  candidates=[ROOT.parent/x/'publication/stage_publication.py' for x in ('strata-0156-tuning-20261103','strata-0156-tuning')]
  path=next((p for p in candidates if p.is_file()),None)
 if path is None:raise Refusal('supply the prior stock publication normalizer')
 s=importlib.util.spec_from_file_location('prior_stock_normalizer',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 return m,path

def private_pins():
 # Source-only extraction: no actual configuration, credential, or process file is read.
 out={}
 for path in (ROOT/'live_benchmark.py',ROOT/'live_benchmark-r1.py',ROOT/'live_benchmark-r2-draft.py'):
  if not path.is_file():continue
  for node in ast.parse(path.read_bytes()).body:
   if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXPECTED' for t in node.targets):
    values=ast.literal_eval(node.value)
    for p,role in PIN_ROLES.items():
     v=values.get(p)
     if isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v):out[v]='<'+role+'>'
 if len(set(out.values()))!=5:raise Refusal('all five reviewed private pin roles must be identifiable in source')
 return out

def project(value):
 if isinstance(value,dict):
  out={}
  for key,child in value.items():
   low=key.lower()
   if key in DROP or ('probe' in low and isinstance(child,dict)):continue
   if ('configuration' in low or 'personal' in low or 'baseline_pin' in low) and ('sha256' in low or 'hash' in low):continue
   if key=='source' and isinstance(child,str) and '.private.' in child:
    out[key]='historical aggregate observation; underlying private production records are excluded';continue
   if key=='model' and isinstance(child,str) and child.startswith('/'):continue
   out[key]=project(child)
  return out
 if isinstance(value,list):return [project(x) for x in value]
 if isinstance(value,str):return RUNTIME.sub('<runtime-path>',value)
 return value

def python_source(path):return path.endswith('.py') or '.py.' in Path(path).name

def encode(N,raw,kind,source,pins):
 if kind in ('report','fixture'):
  data=strict_json(raw)
  if kind=='report':
   if data.get('status') in ('running','starting'):raise Refusal('unfinished report selected: '+source)
   data=project(data)
  raw=(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
 public,counts=N.normalize(raw,Path(source).suffix)
 text=public.decode();text,n=OWNER.subn('<maintenance-owner>',text)
 if n:counts['maintenance_owner']=counts.get('maintenance_owner',0)+n
 for pin,role in pins.items():
  n=text.count(pin)
  if n:text=text.replace(pin,role);counts[role.strip('<>')]=n
 # Preserve Python numeric mock IDs. Report process fields were structurally removed.
 return text.encode(),counts

def validate(relative,raw):
 if python_source(relative):ast.parse(raw.decode(),filename=relative)
 elif relative.endswith('.json'):strict_json(raw)

def select(selection,strict=False):
 found={};missing=[]
 for entry in selection['files']:
  pattern=entry['source']
  if Path(pattern).is_absolute() or '..' in Path(pattern).parts:raise Refusal('selection must be project relative')
  try:paths=sorted(ROOT.glob(pattern))
  except PermissionError:
   if strict:raise
   paths=[]
  if not paths:
   missing.append(pattern)
   if strict and entry.get('required'):raise Refusal('required source missing: '+pattern)
  for path in paths:
   if not path.is_file() or path.is_symlink():raise Refusal('selected source must be a regular file')
   rel=path.relative_to(ROOT).as_posix();target=entry.get('target',rel)
   if any(x in Path(rel).parts for x in ('__pycache__','.stages','.git')) or Path(target).is_absolute() or '..' in Path(target).parts:raise Refusal('unsafe selected path')
   if target in found:raise Refusal('duplicate published target')
   found[target]=(path,entry)
 return found,missing

def check_links(destination,repository):
 for path in destination.rglob('*.md'):
  for target in LINK.findall(path.read_text()):
   target=target.strip().split(' "',1)[0].strip('<>')
   parsed=urllib.parse.urlsplit(target)
   if parsed.scheme or parsed.netloc or not parsed.path:continue
   if parsed.path.startswith('/'):raise Refusal('absolute Markdown file link: '+path.name)
   resolved=(path.parent/urllib.parse.unquote(parsed.path)).resolve()
   try:relative=resolved.relative_to(destination.parents[1])
   except ValueError:raise Refusal('relative link leaves publication repository')
   if resolved.is_file():continue
   if not (repository/relative).is_file():raise Refusal('unresolved relative Markdown link: '+path.name+' -> '+target)

def self_test(N,pins):
 probe={'status':'failed','initial_probe':{'identity':{'pid':123}},'approved_configuration_sha256':{'private':'x'},
  'rounds':[{'status':'contaminated','requests':[{'usage':{'completion_tokens':1024},'actual_content':'omit','response_id':'omit'}],
  'contamination_reasons':['unknown active prompt length'],'solo_observation':{'max_running':1}}],
  'quality':[{'actual':{'checksum':177},'passed':False}],'runtime':'/opt/private/runtime'}
 raw,_=encode(N,json.dumps(probe).encode(),'report','x.json',pins);j=strict_json(raw)
 assert j['status']=='failed' and j['quality'][0]['actual']['checksum']==177
 assert 'initial_probe' not in j and 'actual_content' not in j['rounds'][0]['requests'][0] and '<runtime-path>'==j['runtime']
 code=b'def identity(self,pid=500):\n return pid\n'
 public,_=encode(N,code,'text','mock.py',pins);assert public==code;validate('mock.py',public)
 for pin,role in pins.items():
  public,_=encode(N,('PIN='+repr(pin)+'\n').encode(),'text','pin.py',pins)
  validate('pin.py',public);assert pin.encode() not in public and role.encode() in public
 count=0
 selection=strict_json((ROOT/'publication/SELECTION.json').read_bytes())
 for entry in selection['files']:
  if '*' not in entry['source'] and python_source(entry['source']) and (ROOT/entry['source']).is_file():
   public,_=encode(N,(ROOT/entry['source']).read_bytes(),'text',entry['source'],pins);validate(entry['source'],public);count+=1
 print(json.dumps({'status':'self_test_passed','Python_sources_validated':count,'receipts_read':False,'live_actions':False}))

def execute(args,N,normalizer_path,selection,pins):
 if not args.root_authorized or args.attestation is None:raise Refusal('root publication authority and final attestation required')
 authority_raw=args.attestation.read_bytes();authority=strict_json(authority_raw)
 for key in ('publication_authorized','live_experiment_finished','final_health_verified'):
  if authority.get(key) is not True:raise Refusal('final attestation missing: '+key)
 for name in ('README.md','RESULTS.md'):
  if not (ROOT/name).is_file():raise Refusal('final outcome documentation not supplied: '+name)
 selected,_=select(selection,strict=True)
 if not any(t.startswith('receipts/live-baseline-') and t.endswith('/report.json') for t in selected):raise Refusal('completed live baseline receipt is required')
 if not any(t.startswith('receipts/solo-confidence-') and t.endswith('/report.json') for t in selected):raise Refusal('completed confidence receipt is required')
 destination=ROOT/'publication/.stages'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')/DEST
 destination.mkdir(parents=True,exist_ok=False);records=[];snapshots={}
 for target,(path,entry) in selected.items():
  raw=path.read_bytes();snapshots[path]=sha(raw);public,counts=encode(N,raw,entry['kind'],path.relative_to(ROOT).as_posix(),pins)
  validate(target,public)
  if any(pin.encode() in public for pin in pins):raise Refusal('private pin remains after normalization')
  output=destination/target;output.parent.mkdir(parents=True,exist_ok=True)
  with output.open('xb') as f:f.write(public)
  validate(target,output.read_bytes())
  records.append({'source':path.relative_to(ROOT).as_posix(),'published':target,'source_sha256':sha(raw),'source_bytes':len(raw),
   'published_sha256':sha(public),'published_bytes':len(public),'projection':'structural report privacy filter' if entry['kind']=='report' else 'synthetic fixture' if entry['kind']=='fixture' else 'source/text normalization','redactions':counts})
 check_links(destination,args.repository.resolve())
 if args.attestation.read_bytes()!=authority_raw or any(sha(p.read_bytes())!=h for p,h in snapshots.items()):raise Refusal('authority/source changed during staging')
 manifest={'schema_version':1,'at':dt.datetime.now(dt.timezone.utc).isoformat(),'destination':str(DEST),'files':records,
  'normalizer_source_sha256':sha(normalizer_path.read_bytes()),'root_authority_sha256':sha(authority_raw),
  'private_pin_roles':sorted(PIN_ROLES.values()),'Python_AST_JSON_and_relative_links_validated':True,
  'excluded':selection['excluded'],'repository_mutated':False,'uploaded':False}
 (destination/'PUBLICATION-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 sums=[sha(p.read_bytes())+'  '+p.relative_to(destination).as_posix() for p in sorted(destination.rglob('*')) if p.is_file()]
 (destination/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n')
 print(json.dumps({'status':'staged_local_only','destination':str(destination),'files':len(records),'uploaded':False,'repository_mutated':False}))

def main():
 p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group();m.add_argument('--execute',action='store_true');m.add_argument('--self-test',action='store_true');m.add_argument('--plan',action='store_true')
 p.add_argument('--root-authorized',action='store_true');p.add_argument('--attestation',type=Path);p.add_argument('--normalizer',type=Path);p.add_argument('--repository',type=Path,default=REPO);a=p.parse_args()
 N,np=normalizer(a.normalizer);pins=private_pins();selection=strict_json((ROOT/'publication/SELECTION.json').read_bytes())
 if a.self_test:self_test(N,pins)
 elif a.execute:execute(a,N,np,selection,pins)
 else:
  selected,missing=select(selection);print(json.dumps({'status':'plan_only','destination':str(DEST),'files':list(selected),'missing_or_unreadable_or_pending':missing,'final_outcome_docs_required_before_execution':['README.md','RESULTS.md'],'live_actions':False,'repository_mutated':False},indent=2))
if __name__=='__main__':
 try:main()
 except (Refusal,OSError,ValueError,SyntaxError) as e:print(json.dumps({'status':'refused','error':str(e) if isinstance(e,Refusal) else type(e).__name__}),file=sys.stderr);raise SystemExit(2)
