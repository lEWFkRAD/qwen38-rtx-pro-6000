#!/usr/bin/env python3
"""Bounded private API check of an already native-tested 262K profile."""
import argparse
import copy
import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import urllib.error

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/USER/Documents/strata-0156-tuning-20261103')
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('context_api_helpers', BASE / 'api_canary.py')
API = importlib.util.module_from_spec(spec); spec.loader.exec_module(API)
B = API.CLI
gs = importlib.util.spec_from_file_location('context_preflight_guard', ROOT / 'preflight_guard.py')
G = importlib.util.module_from_spec(gs); gs.loader.exec_module(G)
OWNER = '<maintenance-owner>'
SERVER_SHA = '<private-configuration-sha256>'
CONTRACT_SHA = '<private-configuration-sha256>'


def capacity(h, m, status, slots, prefill):
    for key, value in {'status': 'ok', 'service': 'strata', 'loaded': True, 'max_context': 262144,
                       'images': True, 'api_key': True}.items():
        if h.get(key) != value: raise B.Refusal('private health capacity mismatch: ' + key)
    for key, value in {'batch_slots': slots, 'context': 262144, 'kv': 'int8', 'kv_resident': 0,
                       'expert_slots': 24576, 'mtp_max': 6, 'spec': 8, 'lookup': 3,
                       'engine': '0.1.56', 'prefill_cap': prefill, 'cvec': 'project:4-44'}.items():
        if m.get('engine', {}).get(key) != value: raise B.Refusal('private engine capacity mismatch: ' + key)
    if status.get('context', {}).get('native') != 262144 or status.get('concurrency', {}).get('serving') != slots:
        raise B.Refusal('private advertised capacity differs')
    if status.get('vision', {}).get('available') is not True: raise B.Refusal('matched vision unavailable')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--slots', type=int, choices=(4, 8), required=True)
    p.add_argument('--prefill', type=int, choices=(8192, 16384, 32768), required=True)
    p.add_argument('--native-report', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--long-case', type=Path)
    p.add_argument('--deep-concurrency', action='store_true')
    p.add_argument('--deep-only', action='store_true', help='Only confirm capacity and deep concurrency; earlier feature results remain separate')
    a = p.parse_args()
    if a.deep_only and not a.deep_concurrency: raise B.Refusal('deep-only requires the deep concurrency screen')
    if not a.execute:
        print(json.dumps({'execute': False, 'slots': a.slots, 'context': 262144, 'prefill': a.prefill,
                          'private_port': 18063, 'production_changes': False})); return 0
    prior = json.loads(a.native_report.read_text())
    baseline, contract, cs, hs = B.current_config()
    if (cs, hs) != (SERVER_SHA, CONTRACT_SHA): raise B.Refusal('production baseline changed')
    matches = [arm for arm in prior.get('arms', [prior]) if arm.get('slots') == a.slots and arm.get('prefill') == a.prefill]
    if len(matches) != 1: raise B.Refusal('one matching native arm is required')
    arm = matches[0]
    expected_info = {'context': 262144, 'batch_slots': a.slots, 'prefill_cap': a.prefill,
                     'kv': 'int8', 'kv_resident': 0, 'expert_slots': 24576, 'engine': '0.1.56',
                     'mtp_max': 6, 'spec': 8, 'lookup': 3, 'cvec': 'project:4-44'}
    if (arm.get('infrastructure_status') != 'passed' or arm.get('owned_process_group_empty') is not True
        or arm.get('telemetry', {}).get('errors') or not arm.get('finished_at')
        or arm.get('engine_sha256') != '21cb8dd0c744c84176a019ef3cfebd27997968f211b197429df897cc6af92788'
        or arm.get('selected_profile_sha256') != B.sha(BASE / 'profiles/fused_prefill.json')
        or any(str(arm.get('info', {}).get(k)) != str(v) for k, v in expected_info.items())):
        raise B.Refusal('native evidence does not prove the matching intact resident profile')
    B.engine_args(baseline)
    assert API.CLI.sha(baseline['exe']) == '21cb8dd0c744c84176a019ef3cfebd27997968f211b197429df897cc6af92788'
    cfg = copy.deepcopy(baseline)
    cfg['parallel'] = a.slots
    for flag, value in (('--max-context', 262144), ('--kv-resident', 262144), ('--prefill', a.prefill)):
        cfg['args'] = B.replace_arg(cfg['args'], flag, value)
    settings = {'lib_dirs': baseline['lib_dirs'], 'vision': baseline['vision']['exe'],
                'profile': {'env': {}, 'arg_overrides': {}},
                'library_files': [str(f) for d in baseline['lib_dirs'] for f in Path(d).glob('*.so*') if f.is_file()]}
    manifest = API.runtime_manifest(baseline, contract, Path(baseline['exe']), settings)
    directory = a.run_dir.resolve()
    if not directory.is_relative_to(ROOT): raise B.Refusal('evidence directory must remain in owned project')
    directory.mkdir(mode=0o700, exist_ok=False)
    guard = G.Guard(OWNER, cs, hs, 1200)
    # process_identity validates the candidate arguments through these module-local constants.
    B.SLOTS, B.CAPACITY = a.slots, 262144
    report = {'status': 'running', 'started_at': B.utc(), 'context': 262144, 'slots': a.slots,
              'prefill': a.prefill, 'production_config_sha256': cs, 'production_contract_sha256': hs,
              'manifest': manifest, 'cases': [], 'production_gate_unchanged': True,
              'matching_native_report_sha256': B.sha(a.native_report),
              'private_guard_sha256': B.sha(ROOT / 'preflight_guard.py'),
              'runner_source_sha256': B.sha(Path(__file__)),
              'private_gate_override': 'only owned loopback test wrapper; never production', 'gpu_work_started': False}
    report['deep_only'] = a.deep_only
    wrapper = None
    sampler = B.TelemetrySampler(directory / 'telemetry.jsonl'); sampler.start()
    try:
        wrapper = API.PrivateWrapper(cfg, contract, directory, Path(cfg['exe']), settings, guard, 18063)
        guard.bind_engine(wrapper)
        with wrapper:
            report['gpu_work_started'] = True
            base = 'http://127.0.0.1:18063'
            born = time.monotonic()
            while time.monotonic() - born < 600:
                guard.check(); sampler.check()
                if wrapper.proc.poll() is not None: raise B.Refusal('owned private wrapper exited during startup')
                try:
                    h = API.http_json(base, '/health', wrapper.token, guard, timeout=5)['response']
                    if h.get('loaded') is not True:
                        time.sleep(.5); continue
                    m = API.http_json(base, '/metrics', wrapper.token, guard, timeout=5)['response']
                    s = API.http_json(base, '/v1/status', wrapper.token, guard, timeout=5)['response']
                except (urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout):
                    time.sleep(.5); continue
                report.update(initial_health=h, initial_metrics=m, initial_status=s)
                capacity(h, m, s, a.slots, a.prefill)
                identity = API.process_identity(wrapper.proc, wrapper.ticks, wrapper.private, wrapper.path)
                report.update(initial_health=h, initial_metrics=m, startup_ms=(time.monotonic()-born)*1000,
                              identity=identity)
                break
            else: raise B.Refusal('private API startup deadline expired')
            rejected = API.http_json(base, '/v1/status', None, guard, timeout=10, expected_status=401)
            report['cases'].append({'id': 'unauthenticated_status_rejected', 'status': 'passed', **rejected})
            image = API.make_image(); (directory / 'fresh-square.png').write_bytes(image)
            for name, payload in ([] if a.deep_only else API.request_cases(cfg['model_name'], image)):
                if API.process_identity(wrapper.proc, wrapper.ticks, wrapper.private, wrapper.path) != identity:
                    raise B.Refusal('private process identity changed')
                result = API.http_json(base, '/v1/chat/completions', wrapper.token, guard, payload, timeout=180)
                case = {'id': name, 'status': 'failed', **result}; report['cases'].append(case)
                try:
                    API.validate_completion(name, result['response']); case['status'] = 'passed'
                except API.Refusal as exc: case['failure'] = str(exc)
                print(json.dumps({'event': 'api_case', 'id': name, 'status': case['status']}), flush=True)
            if a.long_case and not a.deep_only:
                case = json.loads(a.long_case.read_text())
                payload = {'model': cfg['model_name'], 'messages': case['messages'], 'max_tokens': case['max_new'],
                           'temperature': 0, 'stream': False, 'reasoning_effort': 'none',
                           'experimental_speed_projection': True, 'chat_template_kwargs': {'enable_thinking': False}}
                result = API.http_json(base, '/v1/chat/completions', wrapper.token, guard, payload, timeout=420)
                text = result['response']['choices'][0]['message'].get('content') or ''
                actual_prompt = result['response'].get('usage', {}).get('prompt_tokens')
                long_record = {'id': 'api_near_capacity_retrieval', 'status': 'failed',
                               'expected': case['expected'], 'prompt_tokens': case['prompt_tokens'],
                               'token_sha256': case['token_sha256'], **result}
                report['cases'].append(long_record)
                if actual_prompt != case['prompt_tokens']:
                    long_record['failure'] = 'prompt token count differs from the frozen native fixture'
                    raise B.Refusal('API prompt token count differs from the frozen native fixture')
                correct = B.check_answer(case, text.strip())
                long_record['status'] = 'passed' if correct else 'failed'
                print(json.dumps({'event':'api_long_case','status':'passed' if correct else 'failed',
                                  'usage':result['response'].get('usage')}),flush=True)
            if a.deep_concurrency:
                deep_spec = importlib.util.spec_from_file_location('context_api_deep_concurrency', ROOT / 'api_deep_concurrency.py')
                deep = importlib.util.module_from_spec(deep_spec); deep_spec.loader.exec_module(deep)
                def identity_check():
                    if API.process_identity(wrapper.proc, wrapper.ticks, wrapper.private, wrapper.path) != identity:
                        raise B.Refusal('private identity changed during deep concurrent requests')
                report['deep_concurrency'] = deep.run(base, wrapper.token, guard, cfg['model_name'],
                    ROOT / 'long-context-deep8-fixtures.json', identity_check, directory, 600)
                if report['deep_concurrency'].get('status') != 'passed':
                    raise B.Refusal('deep API concurrency screen did not pass')
            h = API.http_json(base, '/health', wrapper.token, guard, timeout=15)['response']
            m = API.http_json(base, '/metrics', wrapper.token, guard, timeout=15)['response']
            s = API.http_json(base, '/v1/status', wrapper.token, guard, timeout=15)['response']
            capacity(h, m, s, a.slots, a.prefill)
            if API.process_identity(wrapper.proc, wrapper.ticks, wrapper.private, wrapper.path) != identity:
                raise B.Refusal('private identity changed during requests')
            report.update(final_health=h, final_metrics=m,
                          status='passed' if all(x['status']=='passed' for x in report['cases']) else 'failed')
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
    finally:
        if wrapper:
            wrapper.close()
        report['telemetry'] = sampler.stop()
        report['finished_at'] = B.utc()
        if wrapper:
            report['owned_process_group_empty'] = not wrapper.proc or not API.process_group(wrapper.proc.pid)
            report['private_credential_file_removed'] = not wrapper.path.exists()
        if report['telemetry'].get('errors') or report.get('owned_process_group_empty') is False or report.get('private_credential_file_removed') is False:
            report['status'] = 'failed'
        API.private_json(directory / 'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(directory / 'report.json')}), flush=True)
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    def interrupted(*_): raise KeyboardInterrupt('interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGHUP, interrupted)
    with contextlib.ExitStack() as stack:
        for name in ('.benchmark-engine.lock', '.integrated-engine.lock'):
            lock = stack.enter_context((BASE / name).open('a+'))
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(main())
