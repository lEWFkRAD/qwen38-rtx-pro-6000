"""CPU-only protocol and live-guard tests: no main(), network, GPU or service calls."""
import importlib.util,io,json,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
P=Path(__file__).with_name('live_benchmark.py')
spec=importlib.util.spec_from_file_location('tested_live_benchmark',P);L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)

def chunk(content=None,finish=None,rid='chatcmpl-synthetic',usage=None):
 x={'id':rid,'choices':[{'delta':({'content':content} if content is not None else {}),'finish_reason':finish}]}
 if usage is not None:x['usage']=usage
 return x

def usage(n=512):return {'prompt_tokens':1200,'completion_tokens':n,'total_tokens':1200+n}
def stream(*events,done=True):
 b=b': keepalive\n\n'+b''.join(b'data: '+json.dumps(x).encode()+b'\n\n' for x in events)
 return io.BytesIO(b+(b'data: [DONE]\n\n' if done else b''))

class SSETests(unittest.TestCase):
 def good(self,finish='stop',n=512):return stream(chunk('one'),chunk(' two'),chunk(finish=finish,usage=usage(n)))
 def test_normal_eos_final_usage_not_chunk_count(self):
  with mock.patch.object(L.time,'monotonic',side_effect=[10.,11.,12.]):r=L.read_sse(self.good(),9.)
  self.assertEqual(r['usage']['completion_tokens'],512);self.assertEqual(r['sse_events_observed'],3)
  self.assertEqual(r['actual_content'],'one two');self.assertEqual(r['post_first_token_tokens_s_approximation'],511.)
 def test_length_finish(self):self.assertEqual(L.read_sse(self.good('length'),time.monotonic())['finish_reason'],'length')
 def test_short_normal_eos_is_valid_protocol(self):self.assertEqual(L.read_sse(self.good(n=7),time.monotonic())['usage']['completion_tokens'],7)
 def test_missing_done_rejected(self):
  with self.assertRaises(L.Refusal):L.read_sse(stream(chunk('x'),chunk(finish='stop',usage=usage()),done=False),0)
 def test_changed_response_id_rejected(self):
  with self.assertRaises(L.Refusal):L.read_sse(stream(chunk('x'),chunk('y',rid='other'),chunk(finish='stop',usage=usage())),0)
 def test_missing_response_id_rejected(self):
  with self.assertRaises(L.Refusal):L.read_sse(stream({'choices':[{'delta':{'content':'x'},'finish_reason':'stop'}],'usage':usage()}),0)
 def test_duplicate_usage_rejected(self):
  with self.assertRaises(L.Refusal):L.read_sse(stream(chunk('x',usage=usage()),chunk(finish='stop',usage=usage())),0)
 def test_tool_output_rejected(self):
  x=chunk();x['choices'][0]['delta']={'tool_calls':[{'id':'synthetic'}]}
  with self.assertRaises(L.Refusal):L.read_sse(stream(x),0)
 def test_reasoning_output_rejected(self):
  x=chunk();x['choices'][0]['delta']={'reasoning_content':'synthetic'}
  with self.assertRaises(L.Refusal):L.read_sse(stream(x),0)
 def test_error_event_rejected(self):
  with self.assertRaises(L.Refusal):L.read_sse(stream({'error':{'message':'synthetic'}}),0)
 def test_line_bound(self):
  with self.assertRaises(L.Refusal):L.read_sse(io.BytesIO(b'data: '+b'x'*65540+b'\n'),0)
 def test_inconsistent_total_rejected(self):
  u=usage();u['total_tokens']+=1
  with self.assertRaises(L.Refusal):L.validate_usage(u,1024)
 def test_bool_usage_rejected(self):
  u=usage();u['completion_tokens']=True
  with self.assertRaises(L.Refusal):L.validate_usage(u,1024)
 def test_budget_overrun_rejected(self):
  with self.assertRaises(L.Refusal):L.validate_usage(usage(1025),1024)
 def test_usage_without_visible_output_rejected(self):
  with self.assertRaises(L.Refusal):L.read_sse(stream(chunk(finish='stop',usage=usage())),0)
 def test_arbitrary_metrics_fields_not_exported(self):
  r=L.safe_live({'live':{'state':'generating','request_id':'do-not-export','slots':[{'slot':1,'state':'decoding','generated':2,'prompt_tokens':1200,'request_id':'do-not-export','content':'do-not-export'}]}})
  self.assertNotIn('do-not-export',json.dumps(r))
 def test_negative_or_bool_counter_rejected(self):
  for v in (True,-1,float('nan')):
   with self.assertRaises(L.Refusal):L.counter({'totals':{'requests':v}},'requests')
 def test_payload_sampling_projection_inherited(self):
  p=L.payload(SimpleNamespace(c={'model':'qwen3.8-flash-next'}),{'messages':[{'role':'user','content':'synthetic'}]})
  self.assertNotIn('temperature',p);self.assertNotIn('experimental_speed_projection',p);self.assertEqual(p['max_tokens'],1024)

class GuardTests(unittest.TestCase):
 def guard(self):
  x=object.__new__(L.LiveGuard);x.last=0;x.last_extra=0;x.expected_pins={'expected':'abc'};x.personal_pins={'personal':'xyz'};x.c={};x.latest_memory=None;x.contract=SimpleNamespace(request=mock.Mock());return x
 def test_parent_guard_fails_before_extra_telemetry(self):
  g=self.guard()
  with mock.patch.object(L.P.Guard,'check',side_effect=L.Refusal('base failed')),mock.patch.object(L,'sha') as sha,mock.patch.object(L,'memory') as mem:
   with self.assertRaises(L.Refusal):g.check(force=True)
  sha.assert_not_called();mem.assert_not_called();g.contract.request.assert_not_called()
 def test_bad_pin_fails_before_headroom_or_extra_gets(self):
  g=self.guard()
  with mock.patch.object(L.P.Guard,'check'),mock.patch.object(L,'sha',return_value='wrong'),mock.patch.object(L,'memory') as mem:
   with self.assertRaises(L.Refusal):g.check(force=True)
  mem.assert_not_called();g.contract.request.assert_not_called()
 def test_low_headroom_refuses_before_extra_gets(self):
  for host,gpu in ((6144,1024),(8000,512)):
   g=self.guard()
   with mock.patch.object(L.P.Guard,'check'),mock.patch.object(L,'sha',side_effect=lambda p:{'expected':'abc','personal':'xyz'}[p]),mock.patch.object(L,'memory',return_value={'MemAvailable_mib':host,'gpu_free_mib':gpu}):
    with self.assertRaises(L.Refusal):g.check(force=True)
   g.contract.request.assert_not_called()

class FakeResponse:
 status=200;headers={'Content-Type':'text/event-stream'}
 def __init__(self):self.closed=False;self.stop=threading.Event()
 def close(self):self.closed=True;self.stop.set()
class FakeGuard:
 def __init__(self,fail_after=None):self.started=time.monotonic();self.deadline=self.started+30;self.fail_after=fail_after
 def check(self,force=False):
  if self.fail_after is not None and time.monotonic()-self.started>=self.fail_after:raise L.Refusal('synthetic guard failure')
class FakeContext:
 def __init__(self,state,admissions=True,fail_after=None):
  self.state=state;self.guard=FakeGuard(fail_after);self.hard_deadline=self.guard.deadline;self.c={'base_url':'http://127.0.0.1:8002','model':'qwen3.8-flash-next'};self.server={'api_key':'synthetic-private-token'};self.admissions=admissions
 def metrics(self):
  done=self.state.get('done',False);active=self.state.get('active',False) and not done
  live={'state':'generating' if active else 'idle','running':int(active),'queued':0,'waiting':0,'admission_inflight':int(active and self.admissions),'admitting':0,'outside_slots':int(active),'prompt_tokens':1200 if active else None,'generated':50 if active else None,'slots':[{'slot':i,'state':'idle'} for i in range(8)]}
  return {'live':live,'totals':{'requests':int(done),'output_tokens':512 if done else 0}}

class LifecycleTests(unittest.TestCase):
 def case(self):return {'id':'synthetic','style':'prose','messages':[{'role':'user','content':'synthetic'}],'prompt_tokens':1200}
 def test_own_admission_lease_is_not_external_traffic(self):
  state={};ctx=FakeContext(state,admissions=True);responses=[]
  def open_response(*a,**kw):r=FakeResponse();responses.append(r);return r
  def read_response(r,sent,max_tokens,on_first):
   state['active']=True;on_first();time.sleep(1.1);now=time.monotonic();state['done']=True
   return {'usage':usage(),'timings':None,'finish_reason':'stop','response_id':'chatcmpl-synthetic','actual_content':'synthetic','output_sha256':'x','wall_ms':(now-sent)*1000,'ttft_ms':1,'post_first_token_tokens_s_approximation':None,'post_first_token_rate_scope':'approximate','sse_events_observed':1,'received_bytes':1,'sent_at':sent,'first_at':sent+.001,'last_content_at':now,'completed_at':now}
  with mock.patch.object(L.urllib.request,'build_opener',return_value=SimpleNamespace(open=open_response)),mock.patch.object(L,'read_sse',side_effect=read_response):
   r=L.run_round(ctx,[self.case()],'test',15)
  self.assertFalse(r['contaminated'],r['contamination_reasons']);self.assertTrue(r['owned_http_threads_drained']);self.assertTrue(all(x.closed for x in responses))
 def test_guard_failure_closes_only_owned_handles(self):
  state={};ctx=FakeContext(state,admissions=False,fail_after=.15);responses=[]
  def open_response(*a,**kw):r=FakeResponse();responses.append(r);return r
  def blocked_read(r,sent,max_tokens,on_first):
   state['active']=True;on_first();r.stop.wait(2);state['active']=False;raise OSError('synthetic closed')
  with mock.patch.object(L.urllib.request,'build_opener',return_value=SimpleNamespace(open=open_response)),mock.patch.object(L,'read_sse',side_effect=blocked_read):
   with self.assertRaises(L.Refusal):L.run_round(ctx,[self.case()],'test',15)
  self.assertEqual(len(responses),1);self.assertTrue(responses[0].closed)

if __name__=='__main__':unittest.main()
