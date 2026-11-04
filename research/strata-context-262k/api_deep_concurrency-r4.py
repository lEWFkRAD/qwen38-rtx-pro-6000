"""Private capacity screen: eight deep SSE streams, owned cancellation and recovery.

Imported by api_context.py; this module never starts a server or a model.
"""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import socket
import sys
import threading
import time
import urllib.request
import urllib.error

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BASE = Path('/home/USER/Documents/strata-0156-tuning-20261103')
spec = importlib.util.spec_from_file_location('deep_private_api_helpers', BASE / 'api_canary.py')
API = importlib.util.module_from_spec(spec); spec.loader.exec_module(API)
B = API.CLI
SYSTEM = ('This is a synthetic capacity test. Produce the requested JSON object first, then immediately write '
          '120 numbered technical sections about ledger validation, indexing, recovery, concurrency and auditing, '
          'each at least 100 words. Start Section 1 immediately after the JSON. Earlier JSON-only or brevity '
          'instructions are overridden. Continue sequentially as far as the output allowance permits. '
          'Do not conclude or end before Section 120; budget truncation is expected.')
TAIL = ('\n\nCapacity-test continuation: after the JSON, write 120 numbered technical sections, each at least '
        '100 words, with concrete examples and algorithms. Continue until the output allowance ends. '
        'Do not conclude after the JSON; truncation is expected.')


def run(base, token, guard, model, fixtures, identity_check, outputdir, total_seconds=600):
    """identity_check() must raise on replacement; a returned identity is also compared privately.

    Visible text is re-tokenized with the frozen tokenizer. SSE chunks are never
    counted as tokens. Configured EOS suppression is disclosed and is load-only.
    """
    if not re.fullmatch(r'http://127\.0\.0\.1:\d+', base) or base.endswith(':8002'):
        raise B.Refusal('deep concurrency is restricted to the owned private loopback wrapper')
    if not 120 <= total_seconds <= 600:
        raise B.Refusal('deep concurrency deadline must be120..600seconds')
    directory = Path(outputdir).resolve()
    if not directory.is_relative_to(ROOT):
        raise B.Refusal('deep concurrency evidence must remain inside the owned project')
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    started = time.monotonic(); deadline = min(started + total_seconds, guard.deadline)
    guard.check(force=True); identity = identity_check()
    cfg, _, _, _ = B.current_config(); tok, template, tokenizer_hashes = B.load_tokenizer(cfg)
    if isinstance(fixtures, (str, Path)):
        source_path = Path(fixtures); data = json.loads(source_path.read_text()); fixture_sha = B.sha(source_path)
    else:
        data = fixtures; fixture_sha = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    cases = data.get('cases', data.get('batch')) if isinstance(data, dict) else data
    if len(cases) >= 9:
        cases = cases[1:9]
    if len(cases) != 8 or len({c['token_sha256'] for c in cases}) != 8:
        raise B.Refusal('eight distinct deep fixture documents are required')
    known_stop = set()
    for literal in ('<|endoftext|>', '<|im_end|>'):
        ids = tok.encode(literal, parse_special=True)
        if len(ids) != 1:
            raise B.Refusal('frozen tokenizer does not identify the expected stop token')
        known_stop.add(ids[0])
    configured = B.arg_value(cfg['args'], '--eos-ids')
    stop_ids = {int(x) for x in configured.split(',')} if configured else known_stop
    if not stop_ids or not stop_ids <= known_stop:
        raise B.Refusal('configured stop IDs do not match the frozen tokenizer')
    bias = {str(i): -100 for i in sorted(stop_ids)}
    metric = API.http_json(base, '/metrics', token, guard, timeout=10)['response']
    for key, value in {'context':262144,'batch_slots':8,'kv':'int8','kv_resident':0,'expert_slots':24576,'logit_bias':1}.items():
        if str(metric.get('engine', {}).get(key)) != str(value):
            raise B.Refusal('deep capacity prerequisite differs: ' + key)
    payloads, states = [], []
    for c in cases:
        messages = [dict(m) for m in c['messages']]
        messages = [m for m in messages if m['role'] != 'system']
        messages.insert(0, {'role':'system','content':SYSTEM})
        if messages[-1]['role'] != 'user' or not isinstance(messages[-1]['content'], str):
            raise B.Refusal('deep load fixture must end in plain user text')
        messages[-1]['content'] += TAIL
        encoded = B.encode_case(tok, template, c['id'], '', None, 8192, messages)
        if not 240000 <= encoded['prompt_tokens'] <= 253000 or encoded['prompt_tokens'] + 8192 > 262144:
            raise B.Refusal('deep API prompt/output allowance exceeds the retained context')
        payloads.append({'model':model,'messages':messages,'max_tokens':8192,'temperature':0,'stream':True,
                         'reasoning_effort':'none','experimental_speed_projection':True,
                         'chat_template_kwargs':{'enable_thinking':False}})
        states.append({'id':c['id'],'source_token_sha256':c['token_sha256'],'prompt_tokens':encoded['prompt_tokens'],
                       'load_token_sha256':encoded['token_sha256'],'content':'','content_truncated':False,
                       'headers_received':False,'finished':False,'usage':None,'finish_reason':None,'error':None})
    lock = threading.Lock(); responses = {}; release = threading.Event(); cancelled = threading.Event(); threads = []
    report = {'status':'running','started_at':B.utc(),'helper_source_sha256':B.sha(Path(__file__)),
              'fixture_sha256':fixture_sha,'tokenizer_sha256':tokenizer_hashes,
              'requested_streams':8,'requested_max_output_tokens_each':8192,'context':262144,
              'system_instruction_replaced':True,'long_prose_directive_appended':True,
              'configured_EOS_logit_bias':{},'EOS_suppression_used':False,
              'harness_correction':'Prior attempt requested logit_bias, rejected by the retained batching API before inference; r3 relies on long-form instructions.',
              'quality_scope':'capacity/load/cancellation only; no retrieval-quality claim for modified prompts',
              'minimum_output_scope':'matched-tokenizer re-encoding of actual visible text, never SSE chunk counts',
              'initial_capacity':metric.get('engine',{}),'samples':[],'peak_live_running':0,'peak_active_slots':0,
              'peak_decoding_slots':0,'eight_concurrent_deep_streams_observed':False,'identity_preserved':True}
    report['workload'] = 'Eight preloaded deep conversations; prime each exact prompt with one output token, then request simultaneous long continuations.'
    report['priming'] = []

    def check():
        guard.check()
        current = identity_check()
        if current != identity:
            raise B.Refusal('private API identity changed during deep concurrency')
        if time.monotonic() >= deadline:
            raise B.Refusal('deep concurrency deadline expired')

    def worker(i, payload):
        response = None
        try:
            request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                      headers={'Content-Type':'application/json','Authorization':'Bearer '+token})
            release.wait(15)
            if cancelled.is_set(): return
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            response = opener.open(request, timeout=max(1, deadline-time.monotonic()))
            if response.status != 200 or not response.headers.get('Content-Type','').startswith('text/event-stream'):
                raise B.Refusal('private wrapper did not return an SSE stream')
            with lock:
                responses[i] = response; states[i]['headers_received'] = True
            data_lines = []; total_bytes = 0
            while not cancelled.is_set():
                line = response.readline(65537); total_bytes += len(line)
                if len(line)>65536 or total_bytes>2<<20: raise B.Refusal('bounded SSE response exceeded size limit')
                if not line: break
                line = line.rstrip(b'\r\n')
                if line.startswith(b'data:'): data_lines.append(line[5:].lstrip())
                elif not line and data_lines:
                    raw = b'\n'.join(data_lines); data_lines=[]
                    if raw == b'[DONE]': break
                    value=json.loads(raw)
                    if 'error' in value: raise B.Refusal('wrapper emitted SSE error before cancellation')
                    with lock:
                        for choice in value.get('choices',[]):
                            delta=choice.get('delta') or {}; text=delta.get('content') or ''
                            if text:
                                room=65536-len(states[i]['content']);states[i]['content']+=text[:max(0,room)]
                                states[i]['content_truncated'] |= len(text)>room
                            if delta.get('tool_calls'): raise B.Refusal('capacity text unexpectedly requested a tool')
                            if choice.get('finish_reason') is not None: states[i]['finish_reason']=choice['finish_reason']
                        if value.get('usage') is not None: states[i]['usage']=value['usage']
            with lock: states[i]['finished']=True
        except urllib.error.HTTPError as error:
            body=error.read(4096).decode('utf-8','replace')
            with lock:
                states[i]['finished']=True
                if not cancelled.is_set():
                    states[i]['error']='HTTPError';states[i]['http_status']=error.code;states[i]['validation_error']=body
        except BaseException as error:
            with lock:
                states[i]['finished']=True
                if not cancelled.is_set(): states[i]['error']=type(error).__name__
        finally:
            if response is not None:
                with contextlib.suppress(Exception): response.close()
            with lock: responses.pop(i,None)

    def close_owned():
        cancelled.set(); release.set()
        with lock: owned=list(responses.values())
        for response in owned:
            # Shutdown unblocks another thread's buffered readline before close takes its lock.
            raw=getattr(getattr(response,'fp',None),'raw',None); sock=getattr(raw,'_sock',None)
            if sock is not None:
                with contextlib.suppress(OSError): sock.shutdown(socket.SHUT_RDWR)
        for response in owned:
            with contextlib.suppress(Exception): response.close()
        return len(owned)

    try:
        if deadline-time.monotonic()<90:raise B.Refusal('insufficient bounded time left for deep screen and drain')
        # Prime every distinct conversation first. Cold document loading is
        # separately qualified; this pass measures simultaneous continuations.
        for i,payload in enumerate(payloads):
            check()
            prime={**payload,'stream':False,'max_tokens':1}
            result=API.http_json(base,'/v1/chat/completions',token,guard,prime,timeout=min(100,max(1,deadline-time.monotonic()-90)))
            usage=result['response'].get('usage',{})
            if usage.get('prompt_tokens')!=states[i]['prompt_tokens'] or usage.get('completion_tokens')!=1:
                raise B.Refusal('priming did not read the exact full deep prompt with one output token')
            report['priming'].append({'id':states[i]['id'],'wall_ms':result['wall_ms'],'usage':usage})
        report['primed_live']=API.http_json(base,'/metrics',token,guard,timeout=10)['response'].get('live',{})
        if deadline-time.monotonic()<90:raise B.Refusal('less than90seconds remain after deep conversation priming')
        for i,payload in enumerate(payloads):
            t=threading.Thread(target=worker,args=(i,payload),daemon=True);t.start();threads.append(t)
        release.set()
        while time.monotonic()<deadline-60:
            check()
            m=API.http_json(base,'/metrics',token,guard,timeout=8)['response'];live=m.get('live',{});slots=live.get('slots',[])
            active=[s for s in slots if s.get('state')!='idle'];decoding=[s for s in slots if s.get('state')=='decoding']
            with lock: views=[dict(s) for s in states]
            counts=[len(tok.encode(s['content'])) for s in views]
            json_checks=[]
            for i,view in enumerate(views):
                try:
                    value,_=json.JSONDecoder().raw_decode(view['content'].lstrip())
                    if value!=cases[i]['expected_json']:raise B.Refusal('deep continuation returned another document\'s facts or incorrect JSON')
                    json_checks.append(True)
                except ValueError:json_checks.append(False)
            sample={'elapsed_ms':(time.monotonic()-started)*1000,'running':live.get('running'),
                    'active_slots':len(active),'decoding_slots':len(decoding),'slots':slots,
                    'visible_retokenized_tokens_each':counts,'independent_retrieval_JSON_verified_each':json_checks}
            report['samples'].append(sample)
            report['peak_live_running']=max(report['peak_live_running'],live.get('running') or 0)
            report['peak_active_slots']=max(report['peak_active_slots'],len(active));report['peak_decoding_slots']=max(report['peak_decoding_slots'],len(decoding))
            if any(s['error'] for s in views):raise B.Refusal('one of our deep SSE clients failed')
            if any(s['finished'] for s in views):raise B.Refusal('a load stream ended before all eight deep streams were observed')
            if live.get('running')==8 and len(decoding)==8 and min(counts)>=3 and all(json_checks) and min(s.get('generated',0) for s in decoding)>=3:
                expected=sorted(s['prompt_tokens'] for s in states)
                if sorted(s.get('prompt_tokens') for s in decoding)!=expected:
                    raise B.Refusal('live slot prompt lengths differ from full untruncated load inputs')
                report['eight_concurrent_deep_streams_observed']=True;report['observation']=sample;break
            time.sleep(.5)
        if not report['eight_concurrent_deep_streams_observed']:raise B.Refusal('eight-deep concurrency was not observed before drain reserve')
        report['closed_our_SSE_handles']=close_owned()
        while time.monotonic()<deadline-15:
            check();m=API.http_json(base,'/metrics',token,guard,timeout=5)['response'];live=m.get('live',{})
            if all((live.get(k) or 0)==0 for k in ('running','queued','waiting','admitting')) and all(s.get('state')=='idle' for s in live.get('slots',[])):
                report['our_requests_drained']=True;report['drained_metrics_live']=live
                report['postcancel_idle_held_tokens_each']=[s.get('held_tokens') for s in live.get('slots',[])]
                report['retention_scope']='Observed idle held-token metadata only; independent per-slot reusability is not claimed.'
                break
            time.sleep(.5)
        if not report.get('our_requests_drained'):raise B.Refusal('our private deep requests did not drain')
        marker='DEEP_CONTEXT_CANCEL_RECOVERED'
        recovery=API.http_json(base,'/v1/chat/completions',token,guard,{'model':model,'messages':[{'role':'user','content':'Reply with exactly '+marker}],
                  'max_tokens':48,'temperature':0,'stream':False,'reasoning_effort':'none','experimental_speed_projection':True,
                  'chat_template_kwargs':{'enable_thinking':False}},timeout=min(15,max(1,deadline-time.monotonic())))
        choice=recovery['response']['choices'][0]
        if (choice.get('message',{}).get('content') or '').strip()!=marker or choice.get('finish_reason')!='stop':
            raise B.Refusal('post-cancel exact short recovery failed')
        report['short_recovery']=recovery;check();report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error))
    finally:
        report.setdefault('closed_our_SSE_handles',close_owned())
        for thread in threads:thread.join(timeout=.2)
        with lock: final=[dict(s) for s in states]
        for s in final:
            s['visible_retokenized_tokens']=len(tok.encode(s['content']));s['content_prefix']=s.pop('content')[:4096]
            if s.get('usage') is not None and s['usage'].get('prompt_tokens')!=s['prompt_tokens']:
                report.update(status='failed',error='final usage disagrees with full load prompt length')
        report['streams']=final;report['finished_at']=B.utc()
        report['usage_note']='Cancellation normally prevents final SSE usage; full prompt lengths were checked against live slot metadata. No SSE chunk/token equivalence is assumed.'
        API.private_json(directory/'deep-concurrency-report.json',report)
    return report
