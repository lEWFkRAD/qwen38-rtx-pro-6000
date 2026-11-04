#!/usr/bin/env python3
"""Bounded production feature canary. Default plan performs no requests.

Uses the installed contract's normal authenticated, qualified admission gate.
Never creates a private server, changes a service, overrides the gate, or uses
the privileged startup-probe header. Credentials are read only into memory.
"""
from __future__ import annotations

import argparse
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
import urllib.request

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import api_canary as API

HELPER = Path('/usr/local/libexec/onyx-inference-recovery/strata-contract.py')
CONFIG = Path('/etc/onyx-inference/strata.json')
EXPECTED_ENGINE = '21cb8dd0c744c84176a019ef3cfebd27997968f211b197429df897cc6af92788'


class Refusal(RuntimeError):
    pass


def helper():
    spec = importlib.util.spec_from_file_location('production_strata_contract', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def vision_identity(contract, c, server):
    v = server['vision']
    expected = [v['exe'], '--mmproj', v['mmproj'], '--model', v['model']]
    if v.get('gpu'):
        expected.append('--gpu')
    for key, flag in (('threads', '--threads'), ('max_tokens', '--max-tokens'), ('min_tokens', '--min-tokens')):
        if v.get(key):
            expected.extend((flag, str(v[key])))
    found = []
    group = Path('/sys/fs/cgroup/system.slice') / c['unit'] / 'cgroup.procs'
    for pid in group.read_text().split():
        p = Path('/proc') / pid
        try:
            if (p / 'exe').resolve() != Path(v['exe']).resolve():
                continue
            contract._owned_pid(int(pid), c)
            argv = [os.fsdecode(x) for x in (p / 'cmdline').read_bytes().split(b'\0') if x]
            if argv != expected:
                raise Refusal('production vision arguments differ from the selected configuration')
            found.append({'pid': int(pid), 'start_ticks': (p / 'stat').read_text().rsplit(')', 1)[1].split()[19],
                          'exe': str((p / 'exe').resolve())})
        except FileNotFoundError:
            continue
    if len(found) != 1:
        raise Refusal('exactly one matched production vision process is required')
    return found[0]


def safe_snapshot(s, vision, c):
    # Unit Environment, raw configuration, authentication and startup token are deliberately omitted.
    return {'identity': s['identity'], 'vision_identity': vision, 'health': s['health'],
            'normal_gate_qualified': s['health'].get('qualified') is True,
            'ready_receipt_sha256': sha(c['ready_path'])}


class Guard:
    def __init__(self, contract, c, server, seconds):
        self.contract, self.c, self.server = contract, c, server
        self.deadline = time.monotonic() + seconds
        self.config_sha, self.server_sha = sha(CONFIG), sha(c['server_config'])
        self.identity = self.vision = None
        self.last = 0

    def check(self, force=False):
        if time.monotonic() >= self.deadline:
            raise Refusal('production canary deadline expired')
        if not force and time.monotonic() - self.last < 2:
            return
        self.last = time.monotonic()
        if sha(CONFIG) != self.config_sha or sha(self.c['server_config']) != self.server_sha:
            raise Refusal('production configuration changed during canary')
        self.contract.require_no_maintenance()
        snapshot = self.contract.probe(self.c, require_ready=True, allow_starting=False)
        vision = vision_identity(self.contract, self.c, self.server)
        if self.identity is not None and (snapshot['identity'] != self.identity or vision != self.vision):
            raise Refusal('production wrapper, decoder or vision identity changed')
        self.identity, self.vision = snapshot['identity'], vision
        self.snapshot = safe_snapshot(snapshot, vision, self.c)


def request(contract, c, guard, path, payload=None, timeout=120):
    output = queue.Queue(maxsize=1)
    born = time.monotonic()

    def perform():
        try:
            # Normal authenticated request: never readiness_probe=True.
            output.put(contract.request(c, path, payload, timeout=timeout))
        except BaseException as exc:
            output.put(exc)

    guard.check(force=True)
    threading.Thread(target=perform, daemon=True).start()
    while True:
        guard.check()
        remaining = min(born + timeout, guard.deadline) - time.monotonic()
        if remaining <= 0:
            raise Refusal('production HTTP canary deadline expired')
        try:
            value = output.get(timeout=min(1, remaining))
        except queue.Empty:
            continue
        if isinstance(value, BaseException):
            raise Refusal('production HTTP request failed: ' + type(value).__name__)
        guard.check(force=True)
        return {'http_status': 200, 'wall_ms': (time.monotonic() - born) * 1000, 'response': value}


def exact(value, expected):
    choices = value.get('choices') or []
    if len(choices) != 1:
        raise Refusal('exact canary did not return one choice')
    message = choices[0].get('message', {})
    if (message.get('content') or '').strip() != expected or message.get('tool_calls') or choices[0].get('finish_reason') != 'stop':
        raise Refusal('exact canary answer or finish differs')


def prefix_payload(server, model):
    tok, template, _ = API.CLI.load_tokenizer(server)
    filler = '\n'.join(f'Fictional production-cache record {i:05d}, parcel {(i * 104729 + 20261103) % 1000003:06d}.' for i in range(600))
    body = tok.decode(tok.encode(filler)[:3900])
    marker = 'STRATA_0156_PRODUCTION_CACHE_OK'
    case = API.CLI.encode_case(tok, template, 'production_prefix_reuse',
                               body + '\nReply with exactly ' + marker, marker, 48)
    return {'model': model, 'messages': case['messages'], 'max_tokens': 48, 'temperature': 0,
            'stream': False, 'reasoning_effort': 'none', 'experimental_speed_projection': True,
            'chat_template_kwargs': {'enable_thinking': False}}, marker, case['prompt_tokens']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--total-seconds', type=int, default=600)
    cli = parser.parse_args()
    if not 60 <= cli.total_seconds <= 900:
        raise Refusal('production canary deadline must be60..900seconds')
    if not cli.execute:
        print(json.dumps({'mode': 'plan', 'base_url': 'http://127.0.0.1:8002', 'normal_readiness_gate': True,
                          'cases': ['authenticated capacity/readiness', 'auth rejection', 'exact text',
                                    'forced tool without execution', 'reasoning323', 'fresh green image', '~4k prefix reuse'],
                          'inference_started': False}))
        return 0
    # Keep urllib's process-local routing on explicit loopback even if a caller has proxy variables.
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    contract = helper(); c = contract.load_config(CONFIG)
    if c['environment'].get('ONYX_STRATA_READINESS_GATE') != '1' or c['max_streams'] != 8:
        raise Refusal('normal qualified production gate and eight streams are required')
    server = json.loads(Path(c['server_config']).read_text())
    profile_path = ROOT / 'profiles/fused_prefill.json'
    profile = json.loads(profile_path.read_text())
    selection = json.loads((ROOT / 'FINAL-SELECTION.json').read_text())
    if sha(profile_path) != selection['profile_sha256'] or any(
        str(server.get('env', {}).get(k)) != str(v) for k, v in profile['env'].items()
    ):
        raise Refusal('production environment does not match the frozen selected profile')
    if any(k.startswith('STRATA_LSA') for env in (c['environment'], server.get('env', {})) for k in env):
        raise Refusal('LSA must be absent from the production canary')
    if sha(server['exe']) != EXPECTED_ENGINE:
        raise Refusal('production decoder is not the validated official0.1.56 build')
    contract.verify_static(c)
    guard = Guard(contract, c, server, cli.total_seconds); guard.check(force=True)
    directory = (cli.output_dir or ROOT / 'promotion-20261103' / ('canary-' + str(int(time.time())))).resolve()
    if not directory.is_relative_to(ROOT):
        raise Refusal('production evidence must remain inside the owned Documents project')
    directory.mkdir(parents=True, mode=0o700, exist_ok=False); directory.chmod(0o700)
    report = {'status': 'running', 'started_at': API.CLI.utc(), 'normal_gate_preserved': True,
              'configuration_sha256': guard.server_sha, 'contract_sha256': guard.config_sha,
              'selected_profile_sha256': sha(profile_path),
              'contract_helper_sha256': sha(HELPER), 'initial_probe': guard.snapshot,
              'readiness_override_used': False, 'services_changed': False, 'cases': []}
    try:
        h = request(contract, c, guard, '/health')['response']
        m = request(contract, c, guard, '/metrics')['response']
        s = request(contract, c, guard, '/v1/status')['response']
        API.validate_capacity(h, m, s)
        if m['engine'].get('engine', m['engine'].get('version')) != '0.1.56':
            raise Refusal('production telemetry does not report engine0.1.56')
        report.update(initial_health=h, initial_metrics=m, initial_status=s)
        rejected = API.http_json(c['base_url'], '/v1/status', None, guard, timeout=10, expected_status=401)
        report['cases'].append({'id': 'unauthenticated_status_rejected', 'status': 'passed', **rejected})
        image = API.make_image(); (directory / 'fresh-green-square.png').write_bytes(image)
        report['image'] = {'sha256': hashlib.sha256(image).hexdigest(), 'expected_square_color': 'green'}
        for name, payload in API.request_cases(c['model'], image):
            result = request(contract, c, guard, '/v1/chat/completions', payload, timeout=180)
            case = {'id': name, 'status': 'failed', **result}; report['cases'].append(case)
            try:
                API.validate_completion(name, result['response']); case['status'] = 'passed'
            except API.Refusal:
                case['failure'] = 'synthetic expected response differs'
        payload, marker, count = prefix_payload(server, c['model'])
        prefix = {'id': 'production_prefix_reuse', 'status': 'failed', 'prompt_tokens': count, 'responses': []}
        report['cases'].append(prefix)
        try:
            for _ in range(2):
                value = request(contract, c, guard, '/v1/chat/completions', payload, timeout=180)
                prefix['responses'].append(value); exact(value['response'], marker)
            detail = prefix['responses'][1]['response'].get('usage', {}).get('prompt_tokens_details', {})
            prefix['metadata_available'] = 'cached_tokens' in detail
            prefix['cached_tokens'] = detail.get('cached_tokens')
            if prefix['metadata_available'] and (type(detail['cached_tokens']) is not int or detail['cached_tokens'] <= 0):
                raise Refusal('native usage reported no prefix reuse on the repeated prompt')
            prefix['status'] = 'passed' if prefix['metadata_available'] else 'unavailable'
        except Refusal as exc:
            prefix['failure'] = str(exc)
        guard.check(force=True); contract.verify_static(c); report['final_probe'] = guard.snapshot
        report['final_health'] = request(contract, c, guard, '/health')['response']
        report['final_metrics'] = request(contract, c, guard, '/metrics')['response']
        report['final_status'] = request(contract, c, guard, '/v1/status')['response']
        API.validate_capacity(report['final_health'], report['final_metrics'], report['final_status'])
        report['identity_preserved'] = report['initial_probe']['identity'] == report['final_probe']['identity'] and report['initial_probe']['vision_identity'] == report['final_probe']['vision_identity']
        report['status'] = 'passed' if all(x['status'] in ('passed', 'unavailable') for x in report['cases']) else 'failed'
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=type(exc).__name__)
    finally:
        report['finished_at'] = API.CLI.utc(); API.private_json(directory / 'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(directory / 'report.json')}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    def interrupted(signum, _frame):
        raise KeyboardInterrupt('production canary interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGHUP, interrupted)
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'status': 'refused', 'error': type(exc).__name__}), file=sys.stderr)
        raise SystemExit(2)
