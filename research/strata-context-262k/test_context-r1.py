#!/usr/bin/env python3
"""Bounded isolated262144-token qualification; default is CPU-only plan."""
import argparse
import ast
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import fcntl

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PRIOR = Path('/home/USER/Documents/strata-0156-tuning-20261103')
spec = importlib.util.spec_from_file_location('context_native_adapter', PRIOR / 'benchmark_strata.py')
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
guard_spec = importlib.util.spec_from_file_location('context_preflight_guard', ROOT / 'preflight_guard.py')
P = importlib.util.module_from_spec(guard_spec); guard_spec.loader.exec_module(P)
OWNER = '<maintenance-owner>'
CONTEXT = 262144
ENGINE_SHA = '21cb8dd0c744c84176a019ef3cfebd27997968f211b197429df897cc6af92788'
SERVER_SHA = '<private-configuration-sha256>'
CONTRACT_SHA = '<private-configuration-sha256>'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


class Sampler(B.TelemetrySampler):
    def __init__(self, path):
        super().__init__(path); self.engine = None; self.phase = 'startup'; self.minimum_gpu_free_mib = None

    def _run(self):
        try:
            with self.path.open('x') as stream:
                while not self.stop_event.is_set():
                    try:
                        row = B.telemetry(); row['phase'] = self.phase
                        pid = self.engine.proc.pid if self.engine is not None and self.engine.proc is not None else None
                        row['owned_engine_pid'] = pid
                        apps = B.command(['nvidia-smi', '--id=0', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'])
                        row['compute_apps_csv'] = apps
                        pids = {int(line.split(',')[0].strip()) for line in apps.splitlines() if line.strip()}
                        if pids - ({pid} if pid else set()):
                            self.errors.append('unowned GPU compute process appeared')
                        available = row['mem_available_kib']
                        self.minimum_available_kib = available if self.minimum_available_kib is None else min(available, self.minimum_available_kib)
                        if available < 4 * 1024 * 1024:
                            self.errors.append('host available memory below4GiB')
                        free = float(row['csv'].split(',')[row['fields'].index('memory.free')].strip())
                        self.minimum_gpu_free_mib = free if self.minimum_gpu_free_mib is None else min(free, self.minimum_gpu_free_mib)
                        stream.write(json.dumps(row) + '\n'); stream.flush(); self.samples += 1
                    except Exception as error:
                        self.errors.append(type(error).__name__ + ': ' + str(error))
                    self.stop_event.wait(1)
        except OSError as error:
            self.errors.append(str(error))

    def stop(self):
        result = super().stop(); result['minimum_gpu_free_mib'] = self.minimum_gpu_free_mib; return result


class Engine(B.NativeEngine):
    """Large stdin lines stay interruptible while all normal native parsing is reused."""
    def __init__(self, *args, sampler, **kwargs):
        super().__init__(*args, **kwargs); self.sampler = sampler; self.phase_deadline = self.guard.deadline

    def __enter__(self):
        result = super().__enter__(); os.set_blocking(self.proc.stdin.fileno(), False); return result

    def send(self, line):
        payload = memoryview((line + '\n').encode())
        while payload:
            self.guard.check(); self.sampler.check()
            if time.monotonic() >= min(self.phase_deadline, self.guard.deadline):
                raise B.Refusal('large native request write deadline expired')
            try:
                n = os.write(self.proc.stdin.fileno(), payload[:65536]); payload = payload[n:]
            except BlockingIOError:
                time.sleep(.02)

    def generate(self, case, tok, seconds):
        self.phase_deadline = time.monotonic() + seconds; self.sampler.phase = case['id']
        result = super().generate(case, tok, seconds); self.sampler.check(); return result


def validate(engine, slots, prefill):
    expected = {'context': CONTEXT, 'kv': 'int8', 'kv_resident': 0, 'expert_slots': 24576,
                'batch_slots': slots, 'prefill_cap': prefill, 'spec': 8, 'mtp_max': 6,
                'lookup': 3, 'cvec': 'project:4-44', 'engine': '0.1.56'}
    for key, value in expected.items():
        if str(engine.info.get(key)) != str(value):
            raise B.Refusal('full resident context capacity differs: ' + key)


def batch(engine, cases, tok, seconds):
    n = len(cases); born = time.monotonic(); engine.phase_deadline = born + seconds
    engine.sampler.phase = 'distinct_deep_slot_documents'
    for slot, case in enumerate(cases):
        engine.send(f"BGEN {slot} {case['max_new']} temperature=0 seed=20261103 " + ','.join(map(str, case['tokens'])))
    outputs = {i: [] for i in range(n)}; admissions = {}; admitted = set(); active = set(); completed = {}; current = 0; peak = 0
    while len(completed) < n:
        engine.sampler.check(); when, line = engine.next(born + seconds)
        if line.startswith('T '):
            if current >= n: raise B.Refusal('unassigned native admission token')
            outputs[current].append(int(line.split()[1]))
        elif line.startswith('DONE '):
            if current >= n or current in admissions: raise B.Refusal('unexpected native admission DONE')
            admissions[current] = B.done_fields(line)
        elif line.startswith('BADM '):
            _, slot, cont = line.split(); slot, cont = int(slot), int(cont)
            if slot != current or slot in admitted or slot not in admissions: raise B.Refusal('deep slot identity mismatch')
            admitted.add(slot); current += 1
            if cont == 1: active.add(slot); peak = max(peak, len(active))
            elif cont == 0: completed[slot] = {'generated': len(outputs[slot]), 'finish': admissions[slot]['finish'], 'wall_ms': (when-born)*1000}
            else: raise B.Refusal('invalid BADM continuation flag')
        elif line.startswith('BT '):
            _, slot, token = line.split(); slot = int(slot)
            if slot not in active: raise B.Refusal('deep output belongs to an inactive slot')
            outputs[slot].append(int(token))
        elif line.startswith('BDONE '):
            f = line.split(); slot, count = int(f[1]), int(f[2])
            if slot not in active or count != len(outputs[slot]) or f[3] not in ('stop','length'): raise B.Refusal('deep slot output count/finish mismatch')
            completed[slot] = {'generated': count, 'finish': f[3], 'wall_ms': (when-born)*1000}; active.remove(slot)
        # PP, RESUME and REUSED are normal admission metadata, not an error.
    rows = []
    for slot, case in enumerate(cases):
        text = B.scrub_output(tok, outputs[slot]); good = B.check_answer(case, text)
        fresh = admissions[slot].get('read_tokens',case['prompt_tokens']-admissions[slot]['reused_tokens'])
        if admissions[slot]['prompt_tokens'] != case['prompt_tokens'] or admissions[slot]['reused_tokens']>256 or fresh<case['prompt_tokens']-256:
            raise B.Refusal('distinct deep document did not read at least prompt-minus256 fresh tokens')
        rows.append({'slot': slot, 'id': case['id'], 'status': 'passed' if good else 'failed',
                     'expected': case['expected'], 'actual': text, 'output_ids': outputs[slot],
                     'prompt_tokens': case['prompt_tokens'], 'token_sha256': case['token_sha256'],
                     'native_admission': admissions[slot], **completed[slot]})
    return {'infrastructure_status':'passed', 'quality_status':'passed' if all(x['status']=='passed' for x in rows) else 'failed',
            'all_distinct_slots_mostly_new_and_completed':len(admitted)==n, 'small_shared_prefix_allowance_tokens':256,'peak_live_slots':peak,
            'peak_scope':'observed simultaneous active decodes; all slot pools retain their allocated deep capacity after completion',
            'wall_ms':(time.monotonic()-born)*1000, 'slots':rows}


def load_cases(path, cfg):
    data = json.loads(path.read_text()); cases = data['cases']; tok, template, hashes = B.load_tokenizer(cfg)
    for c in cases:
        assert isinstance(c.get('expected'), str) and c.get('expected_json') is not None and c.get('checker')=='json'
        assert len(c['tokens'])==c['prompt_tokens'] and 240000<=c['prompt_tokens']<=260500
        assert c['prompt_tokens']+c['max_new']<=CONTEXT
        assert hashlib.sha256((','.join(map(str,c['tokens']))+'\n').encode()).hexdigest()==c['token_sha256']
    if len(cases)<9 or len({c['token_sha256'] for c in cases})!=len(cases):
        raise B.Refusal('fixture must contain one near260k case plus eight distinct deep252k cases')
    return cases, tok, template, hashes


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--execute',action='store_true'); p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--owner',default=OWNER); p.add_argument('--run-dir',type=Path); p.add_argument('--arm-seconds',type=int,default=1200);p.add_argument('--total-seconds',type=int,default=2400)
    a=p.parse_args()
    if a.owner!=OWNER or not 60<=a.arm_seconds<=1800 or not 120<=a.total_seconds<=3600:raise B.Refusal('bounded context campaign owner/deadline differs')
    cfg,contract,config_sha,contract_sha=B.current_config(); cases,tok,template,tokhash=load_cases(a.fixtures,cfg)
    if config_sha!=SERVER_SHA or contract_sha!=CONTRACT_SHA:raise B.Refusal('production differs from the explicitly frozen promoted baseline')
    if B.sha(cfg['exe'])!=ENGINE_SHA:raise B.Refusal('installed engine differs from validated official build')
    selected=json.loads((PRIOR/'profiles/fused_prefill.json').read_text())
    if any(str(cfg.get('env',{}).get(k))!=str(v) for k,v in selected['env'].items()):raise B.Refusal('fused production profile changed')
    libraries=B.library_manifest(cfg['lib_dirs']); base=list(cfg['args'])
    if any('rope-' in x or 'yarn-' in x for x in base):raise B.Refusal('unexpected positional scaling;262144is native model context')
    schedule=[]
    for slots,prefill in [(4,32768),(8,16384)]:
        args=base
        for flag,value in [('--max-context',CONTEXT),('--kv-resident',CONTEXT),('--batch',slots),('--prefill',prefill)]:args=B.replace_arg(args,flag,value)
        schedule.append({'slots':slots,'prefill':prefill,'args':args})
    plan={'scope':'isolated native262144 qualification; same resident model,INT8,MTP,controlvector,fused profile; no production promotion',
          'production_config_sha256':config_sha,'production_contract_sha256':contract_sha,'engine_sha256':ENGINE_SHA,
          'fixture_sha256':B.sha(a.fixtures),'tokenizer_sha256':tokhash,'fixture_cases':[{'id':c['id'],'prompt_tokens':c['prompt_tokens'],'token_sha256':c['token_sha256']} for c in cases],
          'schedule':schedule,'library_manifest':libraries,'GPU_work_started':False,'limits':'eight-slot arm follows four-slot infrastructure success; model-quality failures remain failures and do not erase the remaining observations',
          'shared_locks':[str(PRIOR/name) for name in ('.benchmark-engine.lock','.integrated-engine.lock')],
          'preflight_guard_sha256':B.sha(ROOT/'preflight_guard.py'),'selected_profile_sha256':B.sha(PRIOR/'profiles/fused_prefill.json')}
    if not a.execute:print(json.dumps(plan,indent=2));return 0
    directory=(a.run_dir or ROOT/('run-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve();directory.mkdir(parents=True,exist_ok=False)
    dump(directory/'plan.json',plan); guard=P.Guard(a.owner,config_sha,contract_sha,a.total_seconds)
    report={**plan,'status':'running','started_at':B.utc(),'arms':[]}
    with contextlib.ExitStack() as stack:
        for name in ('.benchmark-engine.lock','.integrated-engine.lock'):
            lock=stack.enter_context((PRIOR/name).open('a+'))
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for item in schedule:
                arm_dir=directory/('slots'+str(item['slots'])); sampler=Sampler(directory/('slots'+str(item['slots'])+'-telemetry.jsonl'))
                arm={'slots':item['slots'],'prefill':item['prefill'],'context':CONTEXT,'engine_sha256':ENGINE_SHA,
                     'selected_profile_sha256':plan['selected_profile_sha256'],'status':'running','cases':[]};report['arms'].append(arm)
                args=item['args']; env,public=B.launch_env(cfg,contract,{'lib_dirs':cfg['lib_dirs'],'env':{}});arm['environment']=public
                engine=Engine(Path(cfg['exe']),args,env,cfg['cwd'],arm_dir,guard,min(300,a.arm_seconds),sampler=sampler);sampler.engine=engine
                guard.bind_engine(engine)
                arm_born=time.monotonic(); old_deadline=guard.deadline;guard.deadline=min(old_deadline,arm_born+a.arm_seconds)
                sampler.start()
                try:
                    with engine:
                        validate(engine,item['slots'],item['prefill']);arm.update(info=engine.info,startup_ms=engine.ready_ms,loaded_libraries=B.loaded_library_manifest(engine.proc.pid,libraries))
                        short=B.encode_case(tok,template,'short_exact_before','Reply with exactly CONTEXT262_READY.','CONTEXT262_READY',32)
                        arm['cases'].append(engine.generate(short,tok,a.arm_seconds))
                        cold=engine.generate(cases[0],tok,a.arm_seconds);arm['cases'].append(cold)
                        fresh=cold['native'].get('read_tokens',cold['prompt_tokens']-cold['native']['reused_tokens'])
                        if cold['native']['reused_tokens']>256 or fresh<cold['prompt_tokens']-256:raise B.Refusal('nearcontext primary case reused more than the small shared prefix allowance')
                        messages=[*cases[0]['messages'],{'role':'assistant','content':cold['actual']},{'role':'user','content':'Return exactly the same requested JSON object again, using the records above. No prose.'}]
                        reuse=B.encode_case(tok,template,'deep_accepted_prefix_continuation','',cases[0]['expected'],cases[0]['max_new'],messages)
                        reuse.update(checker='json',expected_json=cases[0]['expected_json'],require_reused=True)
                        if reuse['prompt_tokens']+reuse['max_new']>CONTEXT:raise B.Refusal('continuation exceeds answer reserve')
                        continuation=engine.generate(reuse,tok,a.arm_seconds);arm['cases'].append(continuation)
                        if not continuation['prefix_reuse_gate_passed']:raise B.Refusal('deep accepted-prefix reuse absent')
                        arm['deep_slots']=batch(engine,cases[1:1+item['slots']],tok,a.arm_seconds)
                        recovery=B.encode_case(tok,template,'short_recovery_after','Reply with exactly CONTEXT262_RECOVERED.','CONTEXT262_RECOVERED',32)
                        arm['cases'].append(engine.generate(recovery,tok,a.arm_seconds));validate(engine,item['slots'],item['prefill']);sampler.check()
                        arm['quality_status']='passed' if all(x['quality_status']=='passed' for x in arm['cases']) and arm['deep_slots']['quality_status']=='passed' else 'failed'
                        arm['infrastructure_status']='passed'
                    arm['owned_process_group_empty']=engine.owned_process_group_empty
                    if not arm['owned_process_group_empty']:raise B.Refusal('owned process group not empty')
                    if B.library_manifest(cfg['lib_dirs'])!=libraries:raise B.Refusal('installed libraries changed')
                    arm['status']='passed' if arm['quality_status']=='passed' else 'completed_with_quality_failures'
                except BaseException as exc:
                    arm.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',infrastructure_status='failed',error=type(exc).__name__+': '+str(exc));raise
                finally:
                    arm['telemetry']=sampler.stop();arm['finished_at']=B.utc();guard.deadline=old_deadline;dump(directory/('slots'+str(item['slots'])+'-report.json'),arm);dump(directory/'report.json',report)
                if arm['telemetry']['errors']:raise B.Refusal('periodic context telemetry failed')
                guard.check(force=True,before_start=True)
            report['status']='passed' if all(x['quality_status']=='passed' for x in report['arms']) else 'completed_with_quality_failures'
        except BaseException as exc:
            report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=type(exc).__name__+': '+str(exc))
        finally:
            report['finished_at']=B.utc();dump(directory/'report.json',report)
    print(json.dumps({'status':report['status'],'report':str(directory/'report.json')}));return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt('context campaign interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
    try:raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'status':'refused','error':type(exc).__name__+': '+str(exc)}),file=sys.stderr);raise SystemExit(2)
