"""CPU-only migration tests of the separate r2 draft; no active helper imports/edits."""
import hashlib,importlib.util,json,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
ROOT=Path(__file__).resolve().parent
ACTIVE=ROOT/'live_benchmark.py'
DRAFT=ROOT/'live_benchmark-r2-draft.py'
spec=importlib.util.spec_from_file_location('tested_live_migration_r2_draft',DRAFT)
L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
def row(slot,length,generated,state='decoding'):return {'slot':slot,'prompt_tokens':length,'generated':generated,'state':state}

class CorrelationTests(unittest.TestCase):
 def test_exact_original_reading_zero_binds(self):
  c=L.ActivePromptCorrelation({1200});self.assertEqual(c.observe([row(0,1200,0,'reading')]),[]);self.assertEqual(c.bindings,{0:1200})
 def test_offset_one_and_more_than_one_valid(self):
  for offset in (1,2,50,1000):
   with self.subTest(offset=offset):
    c=L.ActivePromptCorrelation({1200});self.assertEqual(c.observe([row(0,1200+offset,offset)]),[]);self.assertEqual(c.bindings,{0:1200})
 def test_foreign_offset_beyond_generated_refused(self):
  c=L.ActivePromptCorrelation({1200});self.assertTrue(c.observe([row(0,1250,49)]));self.assertEqual(c.bindings,{})
 def test_ambiguous_initial_candidate_rejected(self):
  c=L.ActivePromptCorrelation({1200,1300});self.assertTrue(c.observe([row(0,1350,200)]));self.assertEqual(c.bindings,{})
 def test_existing_binding_cannot_be_reassigned_after_growth(self):
  c=L.ActivePromptCorrelation({1200,1300});self.assertEqual(c.observe([row(0,1250,50)]),[]);self.assertEqual(c.bindings[0],1200)
  self.assertEqual(c.observe([row(0,1350,200)]),[]);self.assertEqual(c.bindings[0],1200)
 def test_idle_clears_and_slot_may_rebind_to_other_original(self):
  c=L.ActivePromptCorrelation({1200,1600});self.assertEqual(c.observe([row(0,1200,0)]),[])
  self.assertEqual(c.observe([{'slot':0,'state':'idle'}]),[]);self.assertEqual(c.bindings,{})
  self.assertEqual(c.observe([row(0,1600,0)]),[]);self.assertEqual(c.bindings,{0:1600})
 def test_one_active_slot_per_original(self):
  c=L.ActivePromptCorrelation({1200});self.assertEqual(c.observe([row(0,1200,0)]),[])
  self.assertTrue(c.observe([row(1,1200,0)]));self.assertEqual(c.bindings,{0:1200})
 def test_unbound_promoted_reading_pending_then_decoding_proves_original(self):
  c=L.ActivePromptCorrelation({1200});self.assertEqual(c.observe([row(0,1250,0,'reading')]),[]);self.assertEqual(c.bindings,{});self.assertEqual(c.pending,{0:1250})
  self.assertEqual(c.observe([row(0,1250,64)]),[]);self.assertEqual(c.bindings,{0:1200});self.assertEqual(c.pending,{})
 def test_bound_promoted_reading_preserves_binding(self):
  c=L.ActivePromptCorrelation({1200});c.observe([row(0,1200,0)])
  self.assertEqual(c.observe([row(0,1250,0,'reading')]),[]);self.assertEqual(c.bindings,{0:1200});self.assertEqual(c.pending,{0:1250})
  self.assertEqual(c.observe([row(0,1250,64)]),[]);self.assertEqual(c.bindings,{0:1200});self.assertEqual(c.pending,{})
 def test_low_and_invalid_generated_rejected_for_shifted_decoding(self):
  for value in (-1,0,1,1025,2.0,True,None):
   with self.subTest(generated=value):self.assertTrue(L.ActivePromptCorrelation({1200}).observe([row(0,1202,value)]))
 def test_invalid_slot_identity_rejected(self):
  for slot in (-1,8,True,0.0,None):
   with self.subTest(slot=slot):self.assertTrue(L.ActivePromptCorrelation({1200}).observe([row(slot,1200,0)]))

class Response:
 status=200;headers={'Content-Type':'text/event-stream'}
 def __init__(self,i):self.i=i;self.closed=False;self.stop=threading.Event()
 def close(self):self.closed=True;self.stop.set()
class Guard:
 def __init__(self):self.deadline=time.monotonic()+30
 def check(self,force=False):pass
class Context:
 def __init__(self,state,originals,lengths,generated):
  self.state=state;self.originals=originals;self.lengths=lengths;self.generated=generated
  self.guard=Guard();self.hard_deadline=self.guard.deadline
  self.c={'base_url':'http://127.0.0.1:8002','model':'qwen3.8-flash-next'};self.server={'api_key':'synthetic-test-only'}
 def metrics(self):
  with self.state['lock']:active=set(self.state['active'])-set(self.state['done']);done=len(self.state['done'])
  slots=[{'slot':i,'state':'idle'} for i in range(8)]
  for i in active:slots[i]=row(i,self.lengths[i],self.generated[i])
  return {'live':{'state':'generating' if active else 'idle','running':len(active),'queued':0,'waiting':0,'admission_inflight':len(active),'admitting':0,'outside_slots':0,'prompt_tokens':self.originals[min(active)] if active else None,'generated':64 if active else None,'slots':slots},'totals':{'requests':done,'output_tokens':512*done}}

class ActualRoundTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.active_before=hashlib.sha256(ACTIVE.read_bytes()).hexdigest()
 @classmethod
 def tearDownClass(cls):
  if hashlib.sha256(ACTIVE.read_bytes()).hexdigest()!=cls.active_before:raise AssertionError('Active helper changed while draft-only CPU tests ran')
 def setUp(self):self.responses=[]
 def run_mock_round(self,lengths,generated=None,wrong_final=False):
  originals=[1200,1600][:len(lengths)];generated=generated or [64]*len(lengths)
  state={'lock':threading.Lock(),'active':set(),'done':set()};ctx=Context(state,originals,lengths,generated)
  cases=[{'id':'synthetic-migration-'+str(i),'style':'prose' if i==0 else 'code','messages':[{'role':'user','content':str(i)}],'prompt_tokens':p} for i,p in enumerate(originals)]
  def opener(request,**_kw):
   i=int(json.loads(request.data)['messages'][-1]['content']);response=Response(i);self.responses.append(response);return response
  def reader(response,sent,max_tokens,on_first):
   i=response.i
   with state['lock']:state['active'].add(i)
   on_first();response.stop.wait(1.1);now=time.monotonic()
   with state['lock']:state['done'].add(i)
   p=originals[i]+int(wrong_final)
   return {'usage':{'prompt_tokens':p,'completion_tokens':512,'total_tokens':p+512},'timings':None,'finish_reason':'stop','response_id':'chatcmpl-owned-synthetic-'+str(i),'actual_content':'synthetic','output_sha256':'synthetic','wall_ms':(now-sent)*1000,'ttft_ms':.1,'post_first_token_tokens_s_approximation':None,'sse_events_observed':1,'received_bytes':10,'sent_at':sent,'first_at':sent+.0001,'last_content_at':now,'completed_at':now}
  with mock.patch.object(L.urllib.request,'build_opener',return_value=SimpleNamespace(open=opener)),mock.patch.object(L,'read_sse',side_effect=reader):return L.run_round(ctx,cases,'mocked-migration',15)
 def test_actual_round_offset_one_accepts_original_final_usage(self):
  r=self.run_mock_round([1201],[3]);self.assertFalse(r['contaminated'],r['contamination_reasons']);self.assertTrue(r['eligible_for_speed']);self.assertEqual(r['requests'][0]['usage']['prompt_tokens'],1200);self.assertTrue(all(x.closed for x in self.responses))
 def test_actual_round_offset_greater_than_one_accepts_original_final_usage(self):
  r=self.run_mock_round([1250],[64]);self.assertFalse(r['contaminated']);self.assertTrue(r['eligible_for_speed']);self.assertEqual(r['requests'][0]['usage']['prompt_tokens'],1200)
 def test_actual_C2_smoke_with_distinct_promoted_inputs(self):
  r=self.run_mock_round([1250,1630],[64,64]);self.assertFalse(r['contaminated'],r['contamination_reasons']);self.assertTrue(r['eligible_for_speed']);self.assertEqual(r['concurrent_clients'],2);self.assertEqual([x['usage']['prompt_tokens'] for x in r['requests']],[1200,1600]);self.assertEqual(r['peak_observed_decoding_slots'],2);self.assertTrue(r['owned_http_threads_drained'])
 def test_actual_foreign_length_refused_and_owned_handle_closed(self):
  r=self.run_mock_round([2500],[512]);self.assertTrue(r['contaminated']);self.assertIn('unknown or ambiguous active prompt length',r['contamination_reasons']);self.assertTrue(all(x.closed for x in self.responses))
 def test_actual_shift_beyond_low_generated_refused(self):self.assertTrue(self.run_mock_round([1250],[3])['contaminated'])
 def test_actual_final_API_usage_still_must_be_original(self):
  with self.assertRaises(L.Refusal):self.run_mock_round([1250],[64],wrong_final=True)

if __name__=='__main__':unittest.main()
