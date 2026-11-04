#!/usr/bin/env python3
"""Bounded confidence-only production SOLO screen; default is a CPU-only plan.

Imports the separately reviewed live benchmark's normal readiness/identity guard.
Never starts/stops/reconfigures production, invokes native control, or sets a
startup-probe header. Only its own ordinary authenticated requests are issued.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
FLOORS = (0.3, 0.5, 0.7)
MAX_NEW = 1024
MIN_OUTPUT = 512
CANARY = Path('/home/USER/Documents/strata-0156-tuning-20261103/promotion-20261103/production_canary.py')
CONFIG = Path('/etc/onyx-inference/strata.json')
EXPECTED_CAPACITY = {'engine': '0.1.56', 'context': 262144, 'batch_slots': 8,
                     'kv': 'int8', 'kv_resident': 0, 'expert_slots': 24576,
                     'prefill_cap': 8192, 'spec': 8, 'mtp_max': 6, 'lookup': 3,
                     'cvec': 'project:4-44'}
PROTOCOL_PROVENANCE = {'official_source_commit': '06698c674673008395ba6e971580538d6b1ea6e0',
    'src/program/generate.cpp_sha256': '3ab400c6885981566e49e6a62b34ac9a79916b161382b91d30977cbe9e88aa05'}


class Refusal(RuntimeError):
    pass


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def collector_sources():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('solo_confidence.py', 'live_benchmark.py')}


def module():
    spec = importlib.util.spec_from_file_location('solo_live_benchmark', ROOT / 'live_benchmark.py')
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def schedule():
    # Reverse each style's floor order in round two to balance position/cache age.
    return [{'round': round_n + 1, 'style_index': style, 'floor': floor}
            for round_n, order in enumerate((FLOORS, tuple(reversed(FLOORS))))
            for floor in order for style in range(2)]


def quality_cases():
    return [
        {'id': 'exact_marker', 'messages': [
            {'role': 'system', 'content': 'Follow the requested exact output format.'},
            {'role': 'user', 'content': 'Reply with exactly SOLO_CONFIDENCE_20261104_OK and nothing else.'}],
         'checker': 'text', 'expected': 'SOLO_CONFIDENCE_20261104_OK'},
        {'id': 'independent_code_result', 'messages': [
            {'role': 'system', 'content': 'Return only the requested JSON object, without Markdown.'},
            {'role': 'user', 'content': 'Evaluate Python: checksum = 17*19 - 23*7 + 11; '
             'values = sorted(set([7,2,5,7])). Return only JSON with keys checksum and values.'}],
         'checker': 'json', 'expected': {'checksum': 173, 'values': [2, 5, 7]}}
    ]


def choose_fixtures(cases):
    chosen, styles = [], set()
    for case in cases:
        style = case.get('style')
        if not isinstance(style, str) or style in styles:
            continue
        if not 1000 <= case.get('prompt_tokens', 0) <= 4500:
            continue
        if not isinstance(case.get('messages'), list) or not case.get('token_sha256'):
            continue
        chosen.append(case); styles.add(style)
        if len(chosen) == 2:
            break
    if len(chosen) != 2 or not any('code' in c['style'].lower() for c in chosen) or not any(
            'prose' in c['style'].lower() for c in chosen):
        raise Refusal('require one frozen roughly 1–4K prose case and one code case')
    if len({c['token_sha256'] for c in chosen}) != 2:
        raise Refusal('confidence styles must use distinct token sequences')
    return chosen


def freeze_fixtures(L, server, seed, path):
    generated = {'schema_version': 1, 'seed': seed, 'kind': 'fresh fictional solo confidence inputs',
                 'cases': choose_fixtures(L.make_fixtures(server, seed=seed)), 'quality': quality_cases()}
    # The fixture file is immutable; reuse is explicit and not a fresh heldout claim.
    if path.exists():
        existing = json.loads(path.read_bytes())
        if digest(existing) != digest(generated):
            raise Refusal('frozen confidence fixtures differ; use a new fixture destination')
        return existing, False
    write_json(path, generated)
    return generated, True


def validate_capacity(metrics):
    engine = metrics.get('engine', {})
    if any(engine.get(k) != v for k, v in EXPECTED_CAPACITY.items()):
        raise Refusal('production capacity, MTP or projection differs from the fixed profile')
    if engine.get('images') is not True:
        raise Refusal('matched production vision must remain available')


def idle_reasons(metrics):
    live = metrics.get('live', {})
    reasons = []
    if live.get('state') != 'idle':
        reasons.append('production_not_idle')
    for key in ('running', 'queued', 'waiting', 'admitting', 'admission_inflight'):
        if live.get(key, 0) != 0:
            reasons.append('nonzero_' + key)
    if any(row.get('state') != 'idle' for row in live.get('slots', [])):
        reasons.append('active_native_slot')
    return reasons


def total(metrics, name):
    value = metrics.get('totals', {}).get(name)
    if type(value) is not int or value < 0:
        raise Refusal('required production total counter is unavailable')
    return value


def require_idle(ctx):
    metrics = ctx.metrics()
    validate_capacity(metrics)
    reasons = idle_reasons(metrics)
    if reasons:
        raise Refusal('solo comparison refused while other activity is observed')
    return metrics


def exact_quality(response, case):
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        return False, None
    choice = choices[0]; message = choice.get('message') or {}
    actual = message.get('content')
    if not isinstance(actual, str) or message.get('tool_calls') or choice.get('finish_reason') != 'stop':
        return False, None
    if case['checker'] == 'text':
        return actual.strip() == case['expected'], actual.strip()
    def no_constant(_):
        raise ValueError('nonfinite JSON')
    def no_duplicates(items):
        output = {}
        for key, value in items:
            if key in output:
                raise ValueError('duplicate JSON key')
            output[key] = value
        return output
    try:
        value = json.loads(actual, parse_constant=no_constant, object_pairs_hook=no_duplicates)
        expected = case['expected']
        passed = (isinstance(value, dict) and set(value) == set(expected)
                  and type(value.get('checksum')) is int and value['checksum'] == expected['checksum']
                  and isinstance(value.get('values'), list) and all(type(x) is int for x in value['values'])
                  and value['values'] == expected['values'])
        return passed, value
    except (ValueError, TypeError):
        return False, actual[:512]


def solo_gate(result):
    reasons = list(result.get('contamination_reasons') or [])
    view = result.get('solo_observation')
    if not isinstance(view, dict):
        return False, sorted(set(reasons + ['solo_observation_missing']))
    for key in ('active_native_slot_observed', 'migration_or_prompt_change_observed'):
        if view.get(key) is not False:
            reasons.append(key)
    for key in ('max_running', 'max_admission_inflight'):
        value = view.get(key)
        if type(value) is not int or not 0 <= value <= 1:
            reasons.append(key)
    for key in ('max_queued', 'max_waiting'):
        if view.get(key) != 0:
            reasons.append(key)
    if view.get('totals_requests_delta') != 1:
        reasons.append('unexpected_completed_request_count')
    if result.get('contaminated') is not False:
        reasons.append('external_activity_or_migration')
    # Sampling cannot prove absence of sub-sample events; disclose that separately.
    return not reasons, sorted(set(reasons))


def numeric_request(row):
    # No response/request IDs, client history, credentials, configuration hashes,
    # process identities or arbitrary model text are written into this report.
    out = {k: row[k] for k in ('usage', 'timings', 'wall_ms', 'ttft_ms', 'finish_reason', 'output_sha256',
                               'post_first_token_tokens_s_approximation') if k in row}
    usage = row.get('usage') or {}; timings = row.get('timings') or {}
    n = usage.get('completion_tokens')
    wall = row.get('wall_ms'); decode = timings.get('predicted_ms')
    out['whole_request_output_tokens_s'] = n * 1000 / wall if type(n) is int and isinstance(wall, (int, float)) and wall > 0 else None
    out['native_decode_output_tokens_s'] = timings.get('predicted_per_second')
    offered, accepted = timings.get('draft_n'), timings.get('draft_n_accepted')
    out['draft_acceptance_fraction'] = accepted / offered if type(offered) is int and type(accepted) is int and offered > 0 else None
    out['minimum_output_gate_passed'] = type(n) is int and n >= MIN_OUTPUT
    rate = out['native_decode_output_tokens_s']
    out['native_decode_timing_available'] = (isinstance(decode, (int, float)) and math.isfinite(decode) and decode > 0
        and isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0)
    out['post_first_scope'] = 'transport approximation from usage and visible-content arrival; not exact per-token decode timing'
    return out


def safe_round(result, case, floor, round_n=None):
    gate, reasons = solo_gate(result)
    rows = result.get('requests') or []
    if len(rows) != 1:
        raise Refusal('solo round did not report exactly one owned request')
    row = numeric_request(rows[0])
    eligible = (gate and result.get('solo_request_tuning_eligible') is True and result.get('eligible_for_speed') is True
                and row['minimum_output_gate_passed'] and row['native_decode_timing_available'])
    return {'id': case['id'], 'style': case['style'], 'floor': floor, 'round': round_n,
            'fixture_prompt_tokens': case['prompt_tokens'], 'fixture_token_sha256': case['token_sha256'],
            'status': result.get('status'), 'eligible_for_speed': eligible,
            'continuing_solo_request_tuning_eligible': result.get('solo_request_tuning_eligible') is True,
            'solo_telemetry_gate_passed': gate, 'contamination_reasons': reasons,
            'solo_observation': result.get('solo_observation'), 'samples': result.get('samples'), 'request': row}


def summarize(rows, quality):
    quality_ok = all(row.get('passed') is True and row.get('external_activity_observed') is False for row in quality) and len(quality) == 6
    cells = []
    for style in sorted({row['style'] for row in rows}):
        for floor in FLOORS:
            selected = [r for r in rows if r['style'] == style and r['floor'] == floor]
            good = len(selected) == 2 and quality_ok and all(r['eligible_for_speed'] for r in selected)
            cells.append({'style': style, 'floor': floor, 'eligible': good,
                'rounds': len(selected),
                'median_native_decode_output_tokens_s': statistics.median(r['request']['native_decode_output_tokens_s'] for r in selected) if good else None,
                'median_whole_request_output_tokens_s': statistics.median(r['request']['whole_request_output_tokens_s'] for r in selected) if good else None})
    comparisons = []
    for style in sorted({row['style'] for row in rows}):
        baseline = next((c for c in cells if c['style'] == style and c['floor'] == .5), None)
        for cell in [c for c in cells if c['style'] == style and c['floor'] != .5]:
            eligible = bool(baseline and baseline['eligible'] and cell['eligible'])
            comparisons.append({'style': style, 'floor': cell['floor'], 'baseline_floor': .5,
                'eligible': eligible,
                'native_decode_change_percent': (cell['median_native_decode_output_tokens_s'] / baseline['median_native_decode_output_tokens_s'] - 1) * 100 if eligible else None,
                'whole_request_change_percent': (cell['median_whole_request_output_tokens_s'] / baseline['median_whole_request_output_tokens_s'] - 1) * 100 if eligible else None})
    return {'all_six_short_quality_cases_passed_without_observed_external_activity': quality_ok,
            'cells': cells, 'comparisons': comparisons,
            'selection_scope': 'two rounds per style; exploratory screen, no automatic production preference or general quality guarantee'}


def remaining(ctx, reserve=20):
    seconds = ctx.guard.deadline - time.monotonic() - reserve
    if seconds < 5:
        raise Refusal('bounded campaign reached cleanup reserve')
    return min(50, seconds)


def execute(L, fixtures, directory, seconds):
    directory.mkdir(parents=True, mode=0o700, exist_ok=False); directory.chmod(0o700)
    report = {'schema_version': 1, 'status': 'running', 'started_at': utc(), 'scope': 'live SOLO confidence-only exploratory screen',
              'normal_readiness_gate': True, 'configuration_changed': False, 'services_changed': False,
              'startup_probe_header_used': False, 'direct_native_control_used': False,
              'request_override': {'strata_tune': {'spec_min_p': list(FLOORS)}},
              'fixed_capacity': EXPECTED_CAPACITY, 'max_output_tokens': MAX_NEW, 'minimum_output_tokens_for_speed': MIN_OUTPUT,
              'fixture_sha256': digest(fixtures), 'quality': [], 'warmups': [], 'timed': [],
              'native_protocol_source': PROTOCOL_PROVENANCE,
              'collector_source_sha256': collector_sources(),
              'limitations': ['confidence affects solo generation; continuing native batch slots use the startup confidence',
                  'other users are not paused; polling plus completed-request counters cannot prove absence of every transient external event',
                  'any observed concurrency, migration or unexpected completed request invalidates the solo comparison',
                  'same prompts are warmed before timing; report actual cache_n and native prompt_n rather than claiming cold results',
                  'normal EOS is retained; outputs below512 tokens are ineligible, never extended with EOS bias',
                  'free-text trajectories may differ; finite exact short checks do not establish general quality equivalence',
                  'SSE chunks are not tokens; post-first arrival metrics are transport approximations']}
    ctx = None
    try:
        ctx = L.load_runtime(seconds=seconds)
        initial = require_idle(ctx); ctx.guard.check(force=True)
        report['initial_capacity_verified'] = True
        for floor in FLOORS:
            for case in fixtures['quality']:
                before = require_idle(ctx)
                payload = {'model': ctx.c['model'], 'messages': case['messages'], 'max_tokens': 64,
                           'stream': False, 'reasoning_effort': 'none', 'chat_template_kwargs': {'enable_thinking': False},
                           'strata_tune': {'spec_min_p': floor}}
                response = ctx.request('/v1/chat/completions', payload, timeout=min(15, remaining(ctx)))['response']
                after = ctx.metrics(); validate_capacity(after)
                external = total(after, 'requests') - total(before, 'requests') != 1 or bool(idle_reasons(after))
                passed, actual = exact_quality(response, case)
                report['quality'].append({'id': case['id'], 'floor': floor, 'passed': passed, 'actual': actual,
                                          'usage': response.get('usage'), 'external_activity_observed': external})
                if external:
                    raise Refusal('external activity invalidated short quality check')
        for case in fixtures['cases']:
            require_idle(ctx)
            result = L.run_round(ctx, [case], 'solo-confidence-warm-' + case['style'], request_seconds=remaining(ctx),
                                 payload_overrides={'strata_tune': {'spec_min_p': .5}}, min_output=MIN_OUTPUT)
            row = safe_round(result, case, .5); report['warmups'].append(row)
            if not row['solo_telemetry_gate_passed']:
                raise Refusal('warmup was not observably solo')
        for item in schedule():
            case = fixtures['cases'][item['style_index']]
            require_idle(ctx)
            label = 'solo-' + case['style'] + '-r' + str(item['round']) + '-p' + str(item['floor'])
            result = L.run_round(ctx, [case], label, request_seconds=remaining(ctx),
                                 payload_overrides={'strata_tune': {'spec_min_p': item['floor']}}, min_output=MIN_OUTPUT)
            row = safe_round(result, case, item['floor'], item['round']); report['timed'].append(row)
            if not row['solo_telemetry_gate_passed']:
                raise Refusal('external activity or native batch migration invalidated solo screen')
        report['analysis'] = summarize(report['timed'], report['quality'])
        report['status'] = 'completed' if all(r['eligible_for_speed'] for r in report['timed']) and report['analysis']['all_six_short_quality_cases_passed_without_observed_external_activity'] else 'ineligible'
    except BaseException as exc:
        report['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        report['error'] = type(exc).__name__
        if isinstance(exc, Refusal):
            report['reason'] = str(exc)
    finally:
        if ctx is not None:
            try:
                final = ctx.metrics(); validate_capacity(final); ctx.guard.check(force=True)
                if collector_sources() != report['collector_source_sha256']:
                    raise Refusal('collector source changed during the confidence campaign')
                report['final_normal_health_and_capacity_verified'] = True
                report['final_idle_observed'] = not idle_reasons(final)
                report['final_counter_delta_requests'] = total(final, 'requests') - total(initial, 'requests') if 'initial' in locals() else None
                if report['final_idle_observed'] is not True:
                    report['status'] = 'contaminated'
            except BaseException as exc:
                report['final_normal_health_and_capacity_verified'] = False
                report['final_check_error'] = type(exc).__name__
                report['status'] = 'failed'
            finally:
                ctx.close()  # shared helper closes only this benchmark's handles/lock.
        report['finished_at'] = utc()
        write_json(directory / 'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(directory / 'report.json')}))
    return 0 if report['status'] == 'completed' else 1


def self_test():
    assert len(schedule()) == 12 and len(quality_cases()) == 2
    for style in (0, 1):
        assert [x['floor'] for x in schedule() if x['style_index'] == style] == [.3, .5, .7, .7, .5, .3]
    case = quality_cases()[1]
    response = lambda text: {'choices': [{'finish_reason': 'stop', 'message': {'content': text}}]}
    assert exact_quality(response('{"checksum":173,"values":[2,5,7]}'), case)[0]
    for wrong in ('{"checksum":173.0,"values":[2,5,7]}', '{"checksum":173,"checksum":173,"values":[2,5,7]}',
                  '{"checksum":173,"values":[2,5,true]}'):
        assert not exact_quality(response(wrong), case)[0]
    result = {'contaminated': False, 'solo_request_tuning_eligible': True, 'contamination_reasons': [], 'solo_observation': {
        'active_native_slot_observed': False, 'migration_or_prompt_change_observed': False,
        'max_running': 1, 'max_admission_inflight': 1, 'max_queued': 0, 'max_waiting': 0, 'totals_requests_delta': 1}}
    assert solo_gate(result)[0]
    result['solo_observation']['active_native_slot_observed'] = True
    assert not solo_gate(result)[0]
    result['solo_observation']['active_native_slot_observed'] = False
    result['solo_observation']['totals_requests_delta'] = 2
    assert not solo_gate(result)[0]
    print(json.dumps({'status': 'self_test_passed', 'network_or_GPU_work': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--fixtures-json', type=Path, default=ROOT / 'solo-confidence-fixtures.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seconds', type=int, default=360)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if not 120 <= args.seconds <= 360:
        raise Refusal('confidence campaign must be bounded to120..360seconds')
    L = module()
    contract = json.loads(CONFIG.read_bytes())
    server = json.loads(Path(contract['server_config']).read_bytes())
    if server.get('sampling') != {'experimental_speed_projection': True}:
        raise Refusal('expected inherited greedy/projection defaults changed; review before benchmarking')
    fixture_path = args.fixtures_json.resolve()
    if not fixture_path.is_relative_to(ROOT):
        raise Refusal('fixture artifact must stay in this Documents project')
    fixtures, fresh = freeze_fixtures(L, server, args.seed, fixture_path)
    plan = {'mode': 'plan', 'inference_started': False, 'normal_readiness_gate': True,
            'fixtures': str(fixture_path), 'fixture_sha256': digest(fixtures), 'newly_frozen': fresh,
            'cases': [{k: c[k] for k in ('id', 'style', 'prompt_tokens', 'token_sha256')} for c in fixtures['cases']],
            'quality_requests': 6, 'warmups': 2, 'maximum_timed_requests': 12, 'output_cap': MAX_NEW,
            'minimum_output_for_speed': MIN_OUTPUT, 'deadline_seconds': args.seconds,
            'request_floor_order': schedule(), 'startup_or_service_changes': False,
            'native_protocol_source': PROTOCOL_PROVENANCE,
            'collector_source_sha256': collector_sources(),
            'scope': 'confidence-only solo screen; one floor wins only if matched quality/eligibility checks pass'}
    if not args.execute:
        print(json.dumps(plan, indent=2)); return 0
    directory = (args.output_dir or ROOT / 'receipts' / ('solo-confidence-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    if not directory.is_relative_to(ROOT):
        raise Refusal('receipt directory must stay in this Documents project')
    return execute(L, fixtures, directory, args.seconds)


if __name__ == '__main__':
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('confidence screen interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGHUP, interrupted)
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'status': 'refused', 'error': type(exc).__name__}), file=sys.stderr)
        raise SystemExit(2)
