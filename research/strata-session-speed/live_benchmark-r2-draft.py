#!/usr/bin/env python3
"""Bounded live production speed baseline; default plan sends no requests.

No service/configuration mutation, private server, startup-probe header, global
cancellation or restarts. SSE final usage supplies counts, never chunk counts.
"""
from __future__ import annotations
import argparse,contextlib,datetime as dt,fcntl,hashlib,importlib.util,json,math,os,queue,random,signal,socket,statistics,subprocess,sys,threading,time,urllib.request
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PRODUCTION_CANARY=Path('/home/USER/Documents/strata-0156-tuning-20261103/promotion-20261103/production_canary.py')
spec=importlib.util.spec_from_file_location('retained_live_production_canary',PRODUCTION_CANARY)
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
B=P.API.CLI
Refusal=P.Refusal
EXPECTED={
 '/etc/onyx-inference/strata-server.json':'<approved-server-config-sha256>',
 '/etc/onyx-inference/strata.json':'<approved-serving-contract-sha256>',
 '/etc/onyx-personal/config.json':'<approved-personal-baseline-sha256>',
 '/usr/local/libexec/onyx-inference-recovery/strata-contract.py':'<approved-contract-helper-sha256>',
 '/etc/onyx-inference/strata-server.shared-settings.json':'<approved-shared-settings-sha256>'}
MAX_BYTES=2<<20
MAX_OUTPUT=1024
MIN_OUTPUT=512

def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):return P.sha(path)
def private_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2);f.write('\n')
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0
def stats(values):
 values=[v for v in values if finite(v)]
 return {'count':len(values),'min':min(values),'median':statistics.median(values),'max':max(values)} if values else {'count':0}
def capacity(h,m,s):
 for k,v in {'status':'ok','service':'strata','loaded':True,'max_context':262144,'images':True,'api_key':True,'qualified':True}.items():
  if h.get(k)!=v:raise Refusal('normal production health mismatch: '+k)
 for k,v in {'engine':'0.1.56','context':262144,'batch_slots':8,'kv':'int8','kv_resident':0,'expert_slots':24576,'prefill_cap':8192,'spec':8,'mtp_max':6,'lookup':3,'cvec':'project:4-44'}.items():
  if m.get('engine',{}).get(k)!=v:raise Refusal('normal production capacity mismatch: '+k)
 if s.get('context',{}).get('native')!=262144 or s.get('concurrency',{}).get('serving')!=8 or s.get('vision',{}).get('available') is not True:
  raise Refusal('normal production context/concurrency/vision mismatch')
def memory():
 mem={}
 for line in Path('/proc/meminfo').read_text().splitlines():
  k,_,v=line.partition(':')
  if k in ('MemAvailable','SwapFree'):mem[k+'_mib']=int(v.strip().split()[0])/1024
 p=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
 values=[float(v.strip()) for v in p.stdout.strip().split(',')]
 if len(values)!=4 or not all(finite(v) for v in values):raise Refusal('invalid numeric GPU telemetry')
 return {**mem,**dict(zip(('gpu_free_mib','gpu_util_percent','gpu_temperature_c','gpu_power_w'),values))}
def safe_live(m):
 live=m.get('live',{});slots=[]
 for x in live.get('slots',[]):
  row={k:x[k] for k in ('slot','prompt_tokens','generated') if finite(x.get(k))}
  row['state']=x.get('state') if x.get('state') in ('idle','reading','decoding','waiting') else 'other';slots.append(row)
 return {'state':live.get('state') if live.get('state') in ('idle','reading','generating','unloaded') else 'other',
  **{k:live.get(k) for k in ('running','queued','waiting','admission_inflight','admitting','outside_slots','prompt_tokens','generated') if finite(live.get(k))},'slots':slots}
def idle(m):
 live=m.get('live',{})
 return all((live.get(k) or 0)==0 for k in ('running','queued','waiting','admission_inflight','admitting')) and live.get('state')=='idle' and all(x.get('state')=='idle' for x in live.get('slots',[]))
def counter(m,k):
 n=m.get('totals',{}).get(k)
 if type(n) is not int or n<0:raise Refusal('invalid production counter: '+k)
 return n

class LiveGuard(P.Guard):
 def __init__(self,contract,c,server,seconds):
  super().__init__(contract,c,server,seconds)
  self.last_extra=0;self.latest_memory=None;self.expected_pins=dict(EXPECTED)
  personal=json.loads(Path('/etc/onyx-personal/config.json').read_text())
  self.personal_pins=dict(personal['baseline_sha256'])
 def check(self,force=False):
  due=force or time.monotonic()-self.last>=2
  super().check(force)
  if due:
   self.last_extra=time.monotonic()
   if any(sha(p)!=v for p,v in self.expected_pins.items()):raise Refusal('approved runtime/personal/shared pin changed')
   if any(sha(p)!=v for p,v in self.personal_pins.items()):raise Refusal('personal baseline pin changed')
   self.latest_memory=memory()
   if self.latest_memory['MemAvailable_mib']<=6144 or self.latest_memory['gpu_free_mib']<=512:
    raise Refusal('live benchmark headroom below host6GiB/GPU512MiB floor')
   h=self.contract.request(self.c,'/health',timeout=3);m=self.contract.request(self.c,'/metrics',timeout=3);s=self.contract.request(self.c,'/v1/status',timeout=3)
   capacity(h,m,s)

class LiveContext:
 def __init__(self,seconds):
  if not 30<=seconds<=480:raise Refusal('live deadline must be30..480seconds including cleanup')
  self.started=time.monotonic();self.hard_deadline=self.started+seconds;self.lock=None
  try:
   self.lock=open(ROOT/'.live-benchmark.lock','a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   self.contract=P.helper();self.c=self.contract.load_config();self.server=json.loads(Path(self.c['server_config']).read_text())
   if self.c['base_url']!='http://127.0.0.1:8002' or self.c['context_length']!=262144 or self.c['max_streams']!=8 or self.c['environment'].get('ONYX_STRATA_READINESS_GATE')!='1':raise Refusal('normal262K production gate required')
   if any(sha(p)!=v for p,v in EXPECTED.items()):raise Refusal('production pins differ from reviewed262K baseline')
   self.contract.verify_static(self.c)
   # Local-module timeout only; never a startup probe or gate bypass.
   self.guard=LiveGuard(self.contract,self.c,self.server,max(1,seconds-20))
   self.contract._readiness_deadline=self.guard.deadline
   self.guard.check(force=True)
  except BaseException:self.close();raise
 def request(self,path,payload=None,timeout=30):return P.request(self.contract,self.c,self.guard,path,payload,timeout)
 def metrics(self):return self.contract.request(self.c,'/metrics',timeout=3)
 def close(self):
  if self.lock is not None:
   fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close();self.lock=None
 def __enter__(self):return self
 def __exit__(self,*_):self.close()
def load_runtime(seconds=480):return LiveContext(seconds)

def make_fixtures(server,seed=20261104):
 tok,template,hashes=B.load_tokenizer(server);rng=random.Random(seed);cases=[]
 for i in range(8):
  style='prose' if i%2==0 else 'code';name=f'synthetic-speed-{seed}-{i}'
  filler='\n'.join(f'Fictional parcel {j:05d}: batch {rng.randrange(1000000):06d}; shelf {rng.randrange(1000):03d}; revision {rng.randrange(1000):03d}.' for j in range(320))
  body=tok.decode(tok.encode(filler)[:1100+i*137])
  task=('Write a detailed operations manual with at least60numbered sections, each with concrete procedures and examples, for this entirely fictional ledger. Cover validation, reconciliation, audit, retention, recovery, reporting, and testing. Start immediately and continue until the allowance ends; do not conclude early.' if style=='prose' else 'Write a complete Python module and many detailed unit tests for this entirely fictional ledger: parsing, validation, filtering, grouping, statistics, JSON export, audit and recovery. Include docstrings and examples. Start immediately and keep implementing tests until the allowance ends; do not conclude early.')
  messages=[{'role':'system','content':'This is a synthetic performance exercise with fictional data. Produce the requested long output. No tools are available.'},{'role':'user','content':f'Independent task {name}.\n{body}\n{task}'}]
  row=B.encode_case(tok,template,name,'',None,MAX_OUTPUT,messages)
  if not 1000<=row['prompt_tokens']<=4000:raise Refusal('synthetic prompt is outside1–4K tokens')
  cases.append({k:row[k] for k in ('id','messages','prompt_tokens','token_sha256')}|{'style':style,'tokenizer_hashes':hashes})
 if len({c['prompt_tokens'] for c in cases})!=8 or len({c['token_sha256'] for c in cases})!=8:raise Refusal('synthetic prompts must have distinct lengths and hashes')
 return cases

def validate_usage(usage,max_tokens):
 if not isinstance(usage,dict):raise Refusal('SSE final usage missing')
 for k in ('prompt_tokens','completion_tokens','total_tokens'):
  if type(usage.get(k)) is not int or usage[k]<0:raise Refusal('invalid SSE usage count')
 if usage['total_tokens']!=usage['prompt_tokens']+usage['completion_tokens'] or usage['completion_tokens']>max_tokens:raise Refusal('SSE usage count violates requested budget')
 return {k:usage[k] for k in ('prompt_tokens','completion_tokens','total_tokens')}|{'prompt_tokens_details':{'cached_tokens':usage.get('prompt_tokens_details',{}).get('cached_tokens')} if isinstance(usage.get('prompt_tokens_details'),dict) else {}}
def read_sse(response,sent,max_tokens=MAX_OUTPUT,on_first=None):
 text=[];data=[];usage=finish=response_id=timings=None;first=last=None;saw_done=False;received=0;events=0
 while True:
  line=response.readline(65537);received+=len(line)
  if len(line)>65536 or received>MAX_BYTES:raise Refusal('SSE response exceeds byte/line bound')
  if not line:break
  line=line.rstrip(b'\r\n')
  if line.startswith(b'data:'):data.append(line[5:].lstrip(b' '));continue
  if line or not data:continue
  raw=b'\n'.join(data);data=[]
  if raw==b'[DONE]':saw_done=True;break
  try:value=json.loads(raw)
  except (ValueError,UnicodeError):raise Refusal('invalid SSE JSON') from None
  if not isinstance(value,dict) or 'error' in value:raise Refusal('SSE error or invalid event')
  events+=1
  if value.get('id') is not None:
   rid=value['id']
   if not isinstance(rid,str) or not 1<=len(rid)<=128 or (response_id is not None and response_id!=rid):raise Refusal('SSE response identity changed')
   response_id=rid
  choices=value.get('choices',[])
  if not isinstance(choices,list) or len(choices)>1:raise Refusal('SSE single-choice contract changed')
  if choices:
   ch=choices[0];delta=ch.get('delta') or {};part=delta.get('content')
   if delta.get('tool_calls') or delta.get('reasoning_content'):raise Refusal('synthetic no-thinking text unexpectedly emitted tools/reasoning')
   if part:
    if not isinstance(part,str):raise Refusal('SSE content must be text')
    now=time.monotonic()
    if first is None:
     first=now
     if on_first:on_first()
    last=now;text.append(part)
   if ch.get('finish_reason') is not None:
    if finish is not None and finish!=ch['finish_reason']:raise Refusal('SSE finish changed')
    finish=ch['finish_reason']
  if value.get('usage') is not None:
   if usage is not None:raise Refusal('duplicate SSE usage')
   usage=validate_usage(value['usage'],max_tokens)
  if value.get('timings') is not None:
   if not isinstance(value['timings'],dict):raise Refusal('invalid SSE timings')
   timings={k:v for k,v in value['timings'].items() if k in ('cache_n','prompt_n','prompt_ms','predicted_n','predicted_ms','predicted_per_second','draft_n','draft_n_accepted') and finite(v)}
 if not saw_done or data or finish not in ('stop','length') or usage is None or response_id is None:raise Refusal('SSE normal finish/usage/DONE/identity missing')
 done=time.monotonic();content=''.join(text)
 if usage['completion_tokens'] and first is None:raise Refusal('usage reports output without visible text')
 approximate=(usage['completion_tokens']-1)/(last-first) if first is not None and last-first>=1 and usage['completion_tokens']>=2 else None
 return {'usage':usage,'timings':timings,'finish_reason':finish,'response_id':response_id,'actual_content':content,'output_sha256':hashlib.sha256(content.encode()).hexdigest(),'wall_ms':(done-sent)*1000,'ttft_ms':None if first is None else (first-sent)*1000,'post_first_token_tokens_s_approximation':approximate,'post_first_token_rate_scope':'Approximation from final usage and first/last visible SSE delivery; batching/buffering prevents exact token arrival timing.','sse_events_observed':events,'received_bytes':received,'sent_at':sent,'first_at':first,'last_content_at':last,'completed_at':done}

def payload(ctx,case,payload_overrides=None):
 out={'model':ctx.c['model'],'messages':case['messages'],'max_tokens':MAX_OUTPUT,'stream':True,'reasoning_effort':'none','chat_template_kwargs':{'enable_thinking':False}}
 if payload_overrides:
  if set(payload_overrides)!={'strata_tune'} or not isinstance(payload_overrides['strata_tune'],dict) or set(payload_overrides['strata_tune'])!={'spec_min_p'} or not finite(payload_overrides['strata_tune']['spec_min_p']) or not 0<=payload_overrides['strata_tune']['spec_min_p']<=1:raise Refusal('only reviewed per-request speculative-confidence override allowed')
  out.update(payload_overrides)
 # Sampling/projection are inherited from the unchanged production configuration.
 return out

class ActivePromptCorrelation:
 """Bound active slots to unique frozen inputs through source-backed promotion.

 Retained server.py1351 sets prompt=list(ids)+out. On decoding,1446 exposes
 cumulative generated=len(out), so 0<=slot_prompt-original<=generated.
 Initial association must be unique; later generated growth never reassigns
 the original. Only an observed idle state clears a binding. Reading an
 already-promoted leg exposes generated0 before1446, so it remains unbound
 pending a decoding observation rather than asserting ownership from no proof.
 Final per-response SSE prompt counts and aggregate ledgers remain exact.
 """
 def __init__(self,initial_lengths):
  self.initials=frozenset(initial_lengths);self.bindings={};self.pending={};self.events=[]
 def observe(self,slots):
  errors=[]
  for row in slots:
   slot=row.get('slot')
   if type(slot) is not int or not 0<=slot<8:
    errors.append('invalid slot identity');continue
   if row.get('state')=='idle':
    if slot in self.bindings:self.events.append({'slot':slot,'event':'idle_binding_cleared','original_prompt_tokens':self.bindings[slot]})
    self.bindings.pop(slot,None);self.pending.pop(slot,None)
  for row in slots:
   slot=row.get('slot');state=row.get('state');length=row.get('prompt_tokens');generated=row.get('generated')
   if state=='idle':continue
   if type(slot) is not int or not 0<=slot<8:continue
   if type(length) is not int or type(generated) is not int or length<1 or not 0<=generated<=MAX_OUTPUT:
    errors.append('invalid slot prompt/generated counters');continue
   prior=self.bindings.get(slot)
   if prior is not None:
    if not 0<=length-prior<=generated:
     # The wrapper clears generated at the start of a promoted read. Keep the
     # existing original binding until decoding can validate the new prefix.
     if state=='reading' and generated==0 and 0<=length-prior<=MAX_OUTPUT:
      self.pending[slot]=length;continue
     errors.append('bound slot prompt violates owned output-count transform');continue
    self.pending.pop(slot,None);continue
   used=set(self.bindings.values())
   candidates=[n for n in self.initials if n not in used and 0<=length-n<=generated]
   if len(candidates)==1:
    original=candidates[0];self.bindings[slot]=original;self.pending.pop(slot,None)
    self.events.append({'slot':slot,'event':'bound','original_prompt_tokens':original,'slot_prompt_tokens':length,'generated_at_binding':generated,'promoted_prefix_tokens':length-original})
   elif state=='reading' and generated==0 and any(0<=length-n<=MAX_OUTPUT for n in self.initials):
    self.pending[slot]=length
   else:errors.append('unknown or ambiguous active prompt length')
  return errors

def run_round(ctx,cases,label,request_seconds=90,payload_overrides=None,min_output=MIN_OUTPUT):
 n=len(cases)
 if n not in (1,2,4,8) or not 15<=request_seconds<=120 or not 1<=min_output<=MAX_OUTPUT:raise Refusal('invalid bounded live round')
 if ctx.guard.deadline-time.monotonic()<request_seconds+5:raise Refusal('insufficient global deadline for this round and drain')
 if len({x['prompt_tokens'] for x in cases})!=n:raise Refusal('distinct synthetic prompt lengths required for count-level traffic detection')
 ctx.guard.check(force=True);before=ctx.metrics()
 if not idle(before):raise Refusal('preexisting production traffic; no synthetic request sent')
 allowed={x['prompt_tokens'] for x in cases};before_requests=counter(before,'requests');before_output=counter(before,'output_tokens');correlation=ActivePromptCorrelation(allowed)
 events=queue.Queue();ready=queue.Queue();release=threading.Event();abort=threading.Event();handles={};lock=threading.Lock();first_seen=set();threads=[]
 contaminated=[];samples=[];completed={};closed_handles=0;all_drained=False
 def note(reason):
  if reason not in contaminated:contaminated.append(reason)
 def close_owned():
  nonlocal closed_handles
  abort.set();release.set()
  with lock:owned=list(handles.values())
  for response in owned:
   raw=getattr(getattr(response,'fp',None),'raw',None);sock=getattr(raw,'_sock',None)
   if sock is not None:
    with contextlib.suppress(OSError):sock.shutdown(socket.SHUT_RDWR)
  for response in owned:
   with contextlib.suppress(Exception):response.close()
  closed_handles+=len(owned)
 def worker(i,case):
  response=None
  try:
   req=urllib.request.Request(ctx.c['base_url']+'/v1/chat/completions',data=json.dumps(payload(ctx,case,payload_overrides)).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+ctx.server['api_key']})
   ready.put(i)
   if not release.wait(10) or abort.is_set():return
   sent=time.monotonic();opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
   response=opener.open(req,timeout=min(10,request_seconds))
   with lock:handles[i]=response
   if abort.is_set():return
   if response.status!=200 or not response.headers.get('Content-Type','').startswith('text/event-stream'):raise Refusal('production did not accept SSE')
   def first():
    with lock:first_seen.add(i)
   result=read_sse(response,sent,MAX_OUTPUT,first)
   if result['usage']['prompt_tokens']!=case['prompt_tokens']:raise Refusal('synthetic prompt count differs from frozen fixture')
   events.put((i,{'id':case['id'],'style':case.get('style'),'prompt_tokens':case['prompt_tokens'],**result}))
  except BaseException as e:events.put((i,{'error':str(e) if isinstance(e,Refusal) else type(e).__name__}))
  finally:
   if response is not None:
    with contextlib.suppress(Exception):response.close()
   with lock:handles.pop(i,None)
 born=time.monotonic();hard=min(ctx.guard.deadline,born+request_seconds);failure=None
 try:
  for i,case in enumerate(cases):
   t=threading.Thread(target=worker,args=(i,case),daemon=True);threads.append(t);t.start()
  prepared=set()
  while len(prepared)<n:
   ctx.guard.check()
   try:prepared.add(ready.get(timeout=.1))
   except queue.Empty:
    if time.monotonic()-born>10:raise Refusal('synthetic clients failed to prepare')
  born=time.monotonic();hard=min(ctx.guard.deadline,born+request_seconds);release.set();next_sample=born
  while len(completed)<n:
   ctx.guard.check()
   if time.monotonic()>=hard:raise Refusal('bounded live round deadline reached')
   try:
    i,row=events.get(timeout=.05)
    if i in completed:raise Refusal('duplicate owned client result')
    if 'error' in row:raise Refusal('owned SSE client failed: '+row['error'])
    completed[i]=row
   except queue.Empty:pass
   if time.monotonic()>=next_sample:
    m=ctx.metrics();view=safe_live(m);active=[x for x in view['slots'] if x['state']!='idle'];decoding=[x for x in active if x['state']=='decoding']
    if (view.get('running') or 0)>n:note('running request count exceeds owned clients')
    for reason in correlation.observe(view['slots']):note(reason)
    if view.get('state') in ('reading','generating') and (view.get('prompt_tokens') is not None and view['prompt_tokens'] not in allowed):note('unknown foreground prompt length')
    with lock:all_started=len(first_seen)==n
    # The normal HTTP gate counts our POST for its entire SSE lifetime. Native
    # queue/admitting counters can also contain our own work; positive is valid.
    if any((view.get(k) or 0)>n for k in ('queued','waiting','admission_inflight','admitting')):note('queue/admission count exceeds owned client population')
    if counter(m,'requests')-before_requests>n:note('unexpected completed-request ledger growth')
    sample={'elapsed_ms':(time.monotonic()-born)*1000,**view,'active_slots':len(active),'decoding_slots':len(decoding),'slot_states':[x['state'] for x in view['slots']],'correlation_pending_reading_slots':sorted(correlation.pending)};samples.append(sample)
    next_sample=time.monotonic()+.25
    if contaminated:raise Refusal('external traffic or ambiguous live activity; stop owned streams')
  for t in threads:t.join(timeout=.1)
 except BaseException as e:failure=e
 finally:
  close_owned()
  drain_end=min(ctx.hard_deadline,time.monotonic()+15)
  for t in threads:t.join(timeout=max(0,min(2,drain_end-time.monotonic())))
  all_drained=all(not t.is_alive() for t in threads)
  # Only GETs are used to settle counters. Never global server cancel or slot control.
  after=None
  while time.monotonic()<drain_end:
   try:
    m=ctx.metrics();after=m
    if counter(m,'requests')-before_requests>=n or (failure is not None and idle(m)):break
   except Exception:break
   time.sleep(.1)
  if not all_drained and failure is None:failure=Refusal('owned HTTP handles did not drain before reserve')
 if failure is not None and not contaminated:raise failure
 if after is None:raise Refusal('post-round production counters unavailable')
 rows=[completed[i] for i in sorted(completed)];total=sum(x['usage']['completion_tokens'] for x in rows)
 requests_delta=counter(after,'requests')-before_requests;output_delta=counter(after,'output_tokens')-before_output
 if len(rows)!=n or requests_delta!=n or output_delta!=total:note('final completed/output counters do not match owned SSE usage')
 if not idle(after):note('production activity remains after owned responses closed')
 ctx.guard.check(force=True)
 ids=[r['response_id'] for r in rows]
 if len(set(ids))!=len(ids):raise Refusal('independent synthetic responses shared an SSE identity')
 ends=[r['completed_at'] for r in rows];duration=max(ends)-born if ends else None
 rates=[r['usage']['completion_tokens']/(r['wall_ms']/1000) for r in rows if r['wall_ms']>0]
 fairness=sum(rates)**2/(len(rates)*sum(x*x for x in rates)) if rates and all(x>0 for x in rates) else None
 peak_decoding=max((s['decoding_slots'] for s in samples),default=0)
 meaningful=all(r['usage']['completion_tokens']>=min_output and r['first_at'] is not None and r['last_content_at']-r['first_at']>=1 for r in rows) and len(rows)==n
 shared_samples=[s['elapsed_ms'] for s in samples if s['decoding_slots']==n]
 shared_span=(shared_samples[-1]-shared_samples[0])/1000 if len(shared_samples)>=2 else 0
 active_native=any(s['active_slots']>0 for s in samples)
 modes={('slots' if s['active_slots'] else 'foreground') for s in samples if (s.get('running') or 0)>0}
 solo={'active_native_slot_observed':active_native,'migration_or_prompt_change_observed':len(modes)>1,'max_running':max((s.get('running') or 0 for s in samples),default=0),'max_queued':max((s.get('queued') or 0 for s in samples),default=0),'max_waiting':max((s.get('waiting') or 0 for s in samples),default=0),'max_admission_inflight':max((max(s.get('admission_inflight') or 0,s.get('admitting') or 0) for s in samples),default=0),'totals_requests_delta':requests_delta,'totals_output_delta':output_delta}
 common_start=max((r['first_at'] for r in rows if r['first_at'] is not None),default=None);common_end=min((r['last_content_at'] for r in rows if r['last_content_at'] is not None),default=None)
 common_seconds=max(0,common_end-common_start) if common_start is not None and common_end is not None else 0
 eligible=meaningful and not contaminated and all_drained and (n==1 or (peak_decoding==n and shared_span>=.5 and common_seconds>=1))
 solo_tuning=n==1 and eligible and not active_native and not solo['migration_or_prompt_change_observed'] and solo['max_running']<=1 and solo['max_admission_inflight']<=1 and all(solo[k]==0 for k in ('max_queued','max_waiting'))
 for r in rows:
  r['sent_after_release_ms']=(r.pop('sent_at')-born)*1000;r['completed_after_release_ms']=(r.pop('completed_at')-born)*1000
  for k in ('first_at','last_content_at'):r.pop(k,None)
 return {'label':label,'status':'contaminated' if contaminated else 'passed' if eligible else 'ineligible','eligible_for_speed':eligible,'concurrent_clients':n,'requested_output_each':MAX_OUTPUT,'minimum_observed_output_each':min_output,'requests':rows,'actual_completion_tokens_total':total,'whole_request_seconds':duration,'whole_request_output_tokens_s':total/duration if duration and duration>0 else None,'per_stream_whole_request_tokens_s':rates,'per_stream_ttft_ms':[r['ttft_ms'] for r in rows],'jain_fairness_from_whole_request_rates':fairness,'release_spread_ms':max((r['sent_after_release_ms'] for r in rows),default=0)-min((r['sent_after_release_ms'] for r in rows),default=0),'peak_observed_decoding_slots':peak_decoding,'all_stream_decoding_samples_span_seconds':shared_span,'samples':samples,'contaminated':bool(contaminated),'contamination_reasons':contaminated,'traffic_correlation_scope':'Stable own SSE response IDs plus counts/unique synthetic prompt lengths; retained wrapper has no SSE-to-history ID join. Undetected same-length, unfinished outside work cannot be ruled out.','common_visible_output_interval_seconds':common_seconds,'exact_common_window_tokens_s':None,'steady_rate_scope':'SSE delivery is buffered; no exact common token rate is inferred. Whole-request final usage is primary; post-first delivery estimates are approximate.','solo_observation':solo,'solo_request_tuning_eligible':solo_tuning,'owned_http_threads_drained':all_drained,'owned_http_handles_closed':closed_handles,'configuration_and_process_identity_preserved':True,'slot_prompt_correlation':{'rule':'original<=slot_prompt_tokens<=original+generated; unique initial association, stable binding until observed idle; promoted reading pending until decoding proof','events':correlation.events,'pending_reading_slots_at_last_sample':sorted(correlation.pending)}}

def main():
 ap=argparse.ArgumentParser(description=__doc__);mode=ap.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true');mode.add_argument('--plan',action='store_true');ap.add_argument('--rounds',type=int,choices=(1,2),default=2);ap.add_argument('--concurrency',type=int,choices=(1,2,4,8));ap.add_argument('--total-seconds',type=int,default=480);ap.add_argument('--request-seconds',type=int,default=75);ap.add_argument('--seed',type=int,default=20261104);ap.add_argument('--output-dir',type=Path)
 a=ap.parse_args()
 if not 60<=a.total_seconds<=480 or not 15<=a.request_seconds<=120:raise Refusal('deadline outside reviewed bounds')
 directory=(a.output_dir or ROOT/'receipts'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
 if not directory.is_relative_to(ROOT):raise Refusal('private baseline evidence must remain in owned project')
 server=json.loads(Path('/etc/onyx-inference/strata-server.json').read_text());cases=make_fixtures(server,a.seed)
 fixture={'seed':a.seed,'synthetic':True,'cases':cases};fixture_hash=hashlib.sha256(json.dumps(fixture,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 order=[(1,[cases[0]]),(4,cases[:4]),(8,cases)]
 if a.rounds==2:order.extend([(8,cases),(4,cases[:4]),(1,[cases[1]])])
 if a.concurrency is not None:
  order=[(a.concurrency,[cases[0]] if a.concurrency==1 else cases[:a.concurrency])]
  if a.rounds==2:order.append((a.concurrency,[cases[1]] if a.concurrency==1 else cases[:a.concurrency]))
 report={'status':'plan','started_at':utc(),'execute':a.execute,'context':262144,'slots':8,'prefill':8192,'ordinary_int8_full24576_vision_required':True,'max_output_each':MAX_OUTPUT,'minimum_output_each_for_speed':MIN_OUTPUT,'round_order':[n for n,_ in order],'synthetic_prompt_tokens':[c['prompt_tokens'] for c in cases],'fixture_sha256':fixture_hash,'source_sha256':{'runner':sha(Path(__file__)),'retained_production_canary':sha(PRODUCTION_CANARY),'installed_contract_helper':sha(P.HELPER)},'scope':'Modest1–4K prompts only; no near-limit eight-deep stress, no new engine/configuration comparison.','sampling':'Inherited production sampling/projection, unchanged cvec; explicit reasoningnone only.','rate_scope':'Final SSE usage / whole request; post-first delivery rates approximate; no chunks-as-tokens.','normal_auth_gate_only':True,'no_config_service_or_server_actions':True,'raw_client_history_or_credentials_saved':False,'rounds':[]}
 directory.mkdir(parents=True,mode=0o700,exist_ok=False);private_json(directory/'fixtures.json',fixture);private_json(directory/'plan.json',report)
 if not a.execute:print(json.dumps({'status':'plan_only','plan':str(directory/'plan.json'),'fixture_sha256':fixture_hash,'round_order':report['round_order']}));return 0
 ctx=None
 try:
  ctx=load_runtime(a.total_seconds);report['initial_probe']=ctx.guard.snapshot;report['initial_headroom']=ctx.guard.latest_memory;report['approved_configuration_sha256']=dict(EXPECTED)
  for index,(n,cs) in enumerate(order):
   row=run_round(ctx,cs,f'round{index+1}-C{n}',a.request_seconds);report['rounds'].append(row)
   print(json.dumps({'round':row['label'],'status':row['status'],'actual_output_tokens':row['actual_completion_tokens_total'],'whole_request_output_tokens_s':row['whole_request_output_tokens_s']}),flush=True)
   if row['contaminated'] or not row['owned_http_threads_drained']:raise Refusal('live round contaminated or not drained; no next round')
  ctx.guard.check(force=True);ctx.contract.verify_static(ctx.c);report['final_probe']=ctx.guard.snapshot;report['final_headroom']=ctx.guard.latest_memory
  report['status']='passed' if all(r['eligible_for_speed'] for r in report['rounds']) else 'completed_with_ineligible_rounds'
 except BaseException as e:report.update(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e) if isinstance(e,Refusal) else type(e).__name__)
 finally:
  if ctx:ctx.close()
  report['finished_at']=utc()
  # Generated text belongs only to our synthetic workloads; summaries retain its hash, not the text.
  for row in report['rounds']:
   for request in row['requests']:request.pop('actual_content',None)
  private_json(directory/'report.json',report)
 print(json.dumps({'status':report['status'],'report':str(directory/'report.json')}));return 0 if report['status'] in ('passed','completed_with_ineligible_rounds') else 1
if __name__=='__main__':
 def stop(*_):raise KeyboardInterrupt('live benchmark interrupted')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGHUP,stop)
 try:raise SystemExit(main())
 except Refusal as e:print(json.dumps({'status':'refused','reason':str(e)}));raise SystemExit(2)
