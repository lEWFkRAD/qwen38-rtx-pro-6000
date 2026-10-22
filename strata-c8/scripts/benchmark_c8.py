#!/usr/bin/env python3
"""Public HTTP-only Strata C8 reproduction harness; no server/config mutations.

Authentication comes from an environment variable. Synthetic output text is
checked in memory; only its SHA-256 and numeric timing data are saved. An idle
start is opportunistic: other requests are never cancelled or blocked.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading
import time
import urllib.error
import urllib.request

DATASET = 'qwen-c8-session-count-v2:distinctcount'


def compare_results(prior, current):
    """Qualify matched output and activity evidence without requiring idle tails."""
    keys = ('dataset', 'model', 'concurrency', 'max_tokens', 'rounds_requested', 'aux_cpus')
    compatible = all(k in prior and k in current and prior[k] == current[k] for k in keys)

    def evidence(result):
        count, concurrency = result.get('rounds_requested'), result.get('concurrency')
        if count not in (1, 2) or concurrency != 8:
            return {}, False, False
        expected = {(r, i) for r in range(1, count + 1) for i in range(concurrency)}
        hashes, complete = {}, True
        rounds = result.get('rounds', [])
        qualified = bool(result.get('summary', {}).get('all_common_windows_qualified')
                         and not result.get('errors') and len(rounds) == count)
        for row in rounds:
            streams = row.get('streams', [])
            qualified &= bool(row.get('common_window', {}).get('valid') and len(streams) == concurrency)
            for stream in streams:
                key = (row.get('round'), stream.get('request'))
                digest = stream.get('sha256')
                valid_hash = isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest) is not None
                complete &= key not in hashes and key in expected and valid_hash
                qualified &= bool(stream.get('checks_passed'))
                hashes[key] = digest if valid_hash else None
        return hashes, bool(complete and set(hashes) == expected), bool(qualified)

    old, old_complete, old_qualified = evidence(prior)
    new, new_complete, new_qualified = evidence(current)
    matches = sum(digest is not None and digest == old.get(key) for key, digest in new.items())
    all_match = bool(compatible and old_complete and new_complete and old == new)
    both = bool(compatible and old_qualified and new_qualified)
    old_rate = prior.get('summary', {}).get('median_session_steady_tokens_per_s')
    new_rate = current.get('summary', {}).get('median_session_steady_tokens_per_s')
    valid_rates = all(isinstance(rate, (int, float)) and not isinstance(rate, bool)
                      and math.isfinite(rate) and rate > 0 for rate in (old_rate, new_rate))
    qualified = bool(both and all_match and valid_rates)
    return {'dataset_compatible': compatible,
            'complete_output_hash_matches': matches, 'complete_outputs_compared': len(new),
            'all_outputs_match': all_match, 'both_common_window_arms_qualified': both,
            'comparison_qualified': qualified,
            'steady_session_gain_percent': (new_rate / old_rate - 1) * 100 if qualified else None}


def result_succeeded(result, comparison_requested=False):
    return bool(result.get('summary', {}).get('all_common_windows_qualified')
                and not result.get('errors')
                and (not comparison_requested or result.get('comparison', {}).get('comparison_qualified')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8002')
    p.add_argument('--model', default='qwen3.8-flash-next-custom-iq3-s')
    p.add_argument('--api-key-env', default='STRATA_API_KEY')
    p.add_argument('--compare', type=Path)
    p.add_argument('--rounds', type=int, choices=(1, 2), default=2)
    p.add_argument('--max-tokens', type=int, choices=(512, 1024), default=512)
    p.add_argument('--aux-cpus', choices=('0', '1', 'default'), default='0')
    p.add_argument('--idle-wait', type=float, default=60)
    p.add_argument('--deadline', type=float, default=180)
    a = p.parse_args()
    if a.output.exists():
        p.error('output exists; choose a fresh filename')
    if not 0 <= a.idle_wait <= 60 or not 30 <= a.deadline <= 300:
        p.error('idle-wait must be 0..60 and deadline 30..300 seconds')
    key = os.environ.get(a.api_key_env)
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    origin, rounds, errors = time.monotonic(), [], []
    deadline = origin + a.deadline

    def now():
        return round(time.monotonic() - origin, 6)

    def budget(cap=5):
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError('overall deadline')
        return min(cap, left)

    def api(route):
        req = urllib.request.Request(a.base_url.rstrip('/') + route, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=budget()) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'HTTP {exc.code}') from None

    def snapshot():
        data = api('/metrics')
        return {'t': now(), 'requests': data.get('totals', {}).get('requests'),
                'live': {k: data.get('live', {}).get(k) for k in (
                    'state', 'running', 'queued', 'waiting', 'admitting', 'outside_slots',
                    'prompt_tokens', 'prompt_total')},
                'engine': {k: data.get('engine', {}).get(k) for k in (
                    'model', 'version', 'context', 'kv', 'kv_resident', 'batch_slots', 'expert_slots')}}

    def idle(s):
        return s['live'].get('state') == 'idle' and all(not s['live'].get(k)
            for k in ('running', 'queued', 'waiting', 'admitting', 'outside_slots'))

    def run_round(index):
        row = {'round': index, 'streams': [], 'samples': []}
        until = min(deadline, time.monotonic() + a.idle_wait)
        while True:
            initial = snapshot()
            if idle(initial):
                break
            if time.monotonic() >= until:
                row.update(skipped=True, full_run_valid=False,
                           common_window={'valid': False}, reasons=['no idle start; no inference sent'])
                return row
            time.sleep(min(0.15, budget(), max(0, until - time.monotonic())))
        row['initial'] = initial
        if initial['engine'].get('batch_slots') != 8:
            raise RuntimeError('expected eight engine slots')
        gate = threading.Barrier(9)

        def stream(i):
            first_number = 1 + 100 * i
            prompt = (f'Benchmark stream {i}. Count the integers from {first_number} through 10000, separated by commas. '
                      'Output only the numbers in ascending order. Continue counting without explanations or abbreviations.')
            payload = {'model': a.model, 'messages': [{'role': 'user', 'content': prompt}],
                       'max_tokens': a.max_tokens, 'temperature': 0, 'stream': True,
                       'stream_options': {'include_usage': True}, 'return_progress': True,
                       'reasoning_effort': 'none', 'experimental_speed_projection': True,
                       'chat_template_kwargs': {'enable_thinking': False}}
            if a.aux_cpus != 'default':
                payload['strata_tune'] = {'aux_cpus': int(a.aux_cpus)}
            gate.wait(timeout=10)
            s = {'request': i, 'first_number': first_number, 'start_t': now(), 'arrivals': []}
            parts, usage, timings, finish = [], {}, None, None
            try:
                req = urllib.request.Request(a.base_url.rstrip('/') + '/v1/chat/completions',
                    headers=headers, data=json.dumps(payload).encode())
                with urllib.request.urlopen(req, timeout=budget()) as response:
                    for line in response:
                        budget()
                        if not line.startswith(b'data:'):
                            continue
                        raw = line[5:].strip()
                        if raw == b'[DONE]':
                            s['done_received'] = True
                            break
                        if not raw:
                            continue
                        chunk = json.loads(raw)
                        if chunk.get('error'):
                            raise RuntimeError('SSE error')
                        usage, timings = chunk.get('usage') or usage, chunk.get('timings') or timings
                        for choice in chunk.get('choices', []):
                            delta = choice.get('delta', {})
                            if delta.get('tool_calls') or delta.get('reasoning_content'):
                                s['unexpected_reasoning_or_tools'] = True
                            text = delta.get('content') or ''
                            if text:
                                parts.append(text)
                                s['arrivals'].append(now())
                            finish = choice.get('finish_reason') or finish
            except Exception as exc:
                s['error'] = type(exc).__name__
            text, times = ''.join(parts), s['arrivals']
            count = usage.get('completion_tokens')
            exact = isinstance(count, int) and len(times) == count
            eos = finish == 'stop' and isinstance(count, int) and len(times) == count - 1
            s.update(end_t=now(), usage=usage, timings=timings, finish_reason=finish,
                     sha256=hashlib.sha256(text.encode()).hexdigest(),
                     one_content_event_per_output_token=exact, one_trailing_stop_token=eos,
                     prefix_passed=re.findall(r'\d+', text)[:40] == list(map(str, range(first_number, first_number + 40))))
            s['ttft_s'] = round(times[0] - s['start_t'], 6) if times else None
            s['decode_elapsed_s'] = round(times[-1] - times[0], 6) if len(times) > 1 else None
            s['content_decode_tokens_per_s'] = round((len(times) - 1) / s['decode_elapsed_s'], 3) if (exact or eos) and s['decode_elapsed_s'] else None
            if timings and timings.get('predicted_n') and timings.get('predicted_ms'):
                s['engine_decode_tokens_per_s'] = round(timings['predicted_n'] * 1000 / timings['predicted_ms'], 3)
            s['checks_passed'] = bool(not s.get('error') and s.get('done_received') and s['prefix_passed']
                and not s.get('unexpected_reasoning_or_tools') and exact and count == a.max_tokens and finish == 'length')
            return s

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(stream, i) for i in range(8)]
            launch_t = now()
            gate.wait(timeout=10)
            while not all(f.done() for f in futures):
                try:
                    row['samples'].append(snapshot())
                except Exception as exc:
                    row['samples'].append({'t': now(), 'error': type(exc).__name__})
                if time.monotonic() >= deadline:
                    break
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            streams = row['streams'] = sorted((f.result() for f in futures), key=lambda s: s['request'])
        final = row['final'] = snapshot()
        correct = all(s['checks_passed'] for s in streams)
        if all(s['arrivals'] for s in streams):
            start = max(s['arrivals'][0] for s in streams) + 0.25
            end = min(s['arrivals'][-1] for s in streams) - 0.25
        else:
            start = end = 0
        samples = [s for s in row['samples'] if start <= s['t'] <= end]
        bad, prompts = [], {s['usage'].get('prompt_tokens') for s in streams}
        for sample in samples:
            if sample.get('error'):
                bad.append({'t': sample['t'], 'reason': 'metrics error'})
                continue
            live = sample['live']
            if (live.get('running') != 8 or any(live.get(k) for k in ('queued', 'waiting', 'admitting', 'outside_slots'))
                or sample['requests'] != initial['requests'] or live.get('state') == 'reading'
                or live.get('prompt_total') not in (None, 0) or live.get('prompt_tokens') not in prompts | {None}):
                bad.append({'t': sample['t'], 'reason': 'activity outside eight own decodes'})
        sample_times = [start] + [s['t'] for s in samples] + [end]
        max_gap = max((b - a for a, b in zip(sample_times, sample_times[1:])), default=0)
        covered = end - start >= 2 and len(samples) >= 3 and 0 <= max_gap <= 0.75
        counts = [sum(start < t <= end for t in s['arrivals']) for s in streams]
        rates = [n / (end - start) for n in counts] if correct and end > start else []
        delta = final['requests'] - initial['requests']
        common_valid = bool(correct and covered and not bad)
        row.update(completed_request_delta=delta,
            full_run_valid=bool(common_valid and delta == 8 and idle(final) and not any(s.get('error') for s in row['samples'])),
            common_window={'valid': common_valid, 'start_t': start, 'end_t': end,
                'duration_s': round(end - start, 6), 'sample_count': len(samples), 'max_sample_gap_s': round(max_gap, 6),
                'observed_contamination': bad, 'per_session_tokens': counts,
                'per_session_tokens_per_s': [round(r, 6) for r in rates],
                'median_session_tokens_per_s': statistics.median(rates) if rates else None,
                'aggregate_tokens_per_s': sum(rates) if rates else None},
            aggregate_wall_tokens_per_s=sum(s['usage'].get('completion_tokens') or 0 for s in streams)
                / (max(s['end_t'] for s in streams) - launch_t))
        return row

    result = {'dataset': DATASET, 'model': a.model, 'concurrency': 8, 'max_tokens': a.max_tokens,
              'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'rounds_requested': a.rounds,
              'aux_cpus': a.aux_cpus, 'rounds': rounds, 'errors': errors}
    try:
        for i in range(1, a.rounds + 1):
            rounds.append(run_round(i))
            a.output.write_text(json.dumps(result, indent=2) + '\n')
            if rounds[-1].get('skipped'):
                break
    except Exception as exc:
        errors.append(type(exc).__name__)
    qualified = [r for r in rounds if r['common_window']['valid']]
    clean = [r for r in rounds if r['full_run_valid']]
    result['summary'] = {'qualified_common_windows': len(qualified), 'fully_clean_rounds': len(clean),
        'all_common_windows_qualified': len(qualified) == a.rounds and not errors,
        'median_session_steady_tokens_per_s': statistics.median(r['common_window']['median_session_tokens_per_s'] for r in qualified) if qualified else None,
        'median_aggregate_steady_tokens_per_s': statistics.median(r['common_window']['aggregate_tokens_per_s'] for r in qualified) if qualified else None,
        'median_aggregate_wall_tokens_per_s': statistics.median(r['aggregate_wall_tokens_per_s'] for r in clean) if clean else None}
    if a.compare:
        try:
            result['comparison'] = compare_results(json.loads(a.compare.read_text()), result)
        except Exception as exc:
            # Error text can contain private paths or endpoints; retain its type only.
            errors.append('Comparison' + type(exc).__name__)
            result['comparison'] = {'comparison_qualified': False, 'steady_session_gain_percent': None}
    result['elapsed_s'] = now()
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'summary': result['summary'], 'comparison': result.get('comparison'), 'errors': errors}))
    if not result_succeeded(result, comparison_requested=a.compare is not None):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
