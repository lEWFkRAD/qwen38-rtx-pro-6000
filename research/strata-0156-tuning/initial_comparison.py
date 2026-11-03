#!/usr/bin/env python3
"""First version-only stock-engine screen using the guarded native adapter."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import benchmark_strata as B

ROOT = Path(__file__).resolve().parent


class MemoryGuard(B.Guard):
    def check(self, force=False, before_start=False):
        super().check(force, before_start)
        if time.monotonic() - getattr(self, 'last_memory_check', 0) >= 2:
            self.last_memory_check = time.monotonic()
            mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            if int(mem['MemAvailable'].split()[0]) < 4 * 1024 * 1024:
                raise B.Refusal('host available memory fell below 4 GiB')


def fixtures(cfg):
    tok, template, token_hashes = B.load_tokenizer(cfg)
    definitions = json.loads((ROOT / 'quality-fixtures.json').read_text())['tuning']
    cases = [B.encode_case(tok, template, item['id'], item['question'], item['expected'], item['max_new'])
             for item in definitions]
    workloads = []
    for slot in range(9):
        name = 'solo' if slot == 8 else f'stream{slot}'
        filler = '\n'.join(f'Synthetic {name} record {i:05d}: parcel {(i * 7919 + (slot + 1) * 104729) % 1000003:06d}; bay {(i * 37 + slot * 11) % 997:03d}.' for i in range(500))
        body = tok.decode(tok.encode(filler)[:3900])
        task = ('Write a complete Python module for this fictional ledger with parsing, validation, filtering, grouping, statistics, JSON export, and at least twenty detailed tests. Include docstrings and examples. Produce at least 150 lines; continue until the whole module is complete.'
                if slot % 2 else 'Write a detailed operational runbook for this fictional ledger with at least 150 numbered steps. Explain validation, reconciliation, retention, error handling, recovery, monitoring, and worked examples. Continue until the complete runbook is written.')
        question = f'Independent workload {name}. All records below are fictional.\n{body}\n{task}'
        workloads.append(B.encode_case(tok, template, name, question, None, 256))
    return tok, token_hashes, cases, workloads[:8], workloads[8]


def telemetry(path, stop):
    query = 'timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,pstate'
    with path.open('x') as stream:
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu='+query, '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            stream.write(json.dumps({'at':B.utc(),'gpu_csv':result.stdout.strip(),'returncode':result.returncode,
                                     'mem_available_kib':int(mem['MemAvailable'].split()[0]),'swap_free_kib':int(mem['SwapFree'].split()[0])})+'\n')
            stream.flush()
            stop.wait(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--owner', default='')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--profiles', nargs='+', default=['official0150_current','stock0156_same_settings'])
    cli = parser.parse_args()
    cfg, contract, config_sha, contract_sha = B.current_config()
    definitions = json.loads((ROOT/'profiles.json').read_text())
    control = dict(definitions['control'])
    control.update(executable=cfg['exe'], sha256=B.sha(cfg['exe']), lib_dirs=cfg['lib_dirs'])
    available = {p['name']:p for p in [control,*definitions['profiles']]}
    selected = [available[name] for name in cli.profiles]
    for profile in selected:
        assert B.sha(profile['executable']) == profile['sha256']
    if not cli.execute:
        print(json.dumps({'scope':'version-only screen; tuning6, actual8stream256, solo256; all8slots131072','profiles':selected,'gpu_work_started':False},indent=2)); return
    if not cli.owner:
        raise B.Refusal('exact owned maintenance owner is required')
    cli.run_dir.mkdir(parents=True,exist_ok=False)
    tok, token_hashes, cases, batch, solo = fixtures(cfg)
    (cli.run_dir/'fixtures.json').write_text(json.dumps({'quality':cases,'batch':batch,'solo':solo,'tokenizer_hashes':token_hashes},indent=2)+'\n')
    report = {'scope':'first stock-engine screen; one pass per profile, no statistical speed claim','started_at':B.utc(),
              'config_sha256':config_sha,'contract_sha256':contract_sha,'status':'running','arms':[]}
    guard = MemoryGuard(cli.owner, config_sha, contract_sha, 1800)
    with (ROOT/'.benchmark-engine.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for profile in selected:
                name=profile['name']; arm={'profile':profile,'cases':[],'started_at':B.utc(),'status':'running'}
                report['arms'].append(arm)
                print(json.dumps({'event':'launch','profile':name}),flush=True)
                env, public = B.launch_env(cfg, contract, profile)
                args = B.apply_profile_args(B.engine_args(cfg,'full'),profile)
                arm.update(args=args,environment=public)
                stop=threading.Event(); sampler=threading.Thread(target=telemetry,args=(cli.run_dir/(name+'-telemetry.jsonl'),stop),daemon=True);sampler.start()
                try:
                    with B.NativeEngine(Path(profile['executable']),args,env,cfg['cwd'],cli.run_dir/name,guard,300) as engine:
                        arm.update(info=engine.info,ready_ms=engine.ready_ms,drafting=B.validate_capacity(engine))
                        for case in cases:
                            value=engine.generate(case,tok,180);arm['cases'].append(value)
                            print(json.dumps({'event':'quality','profile':name,'case':case['id'],'status':value['status']}),flush=True)
                        arm['solo']=engine.generate(solo,tok,180)
                        arm['throughput']=engine.throughput(batch,tok,180)
                        arm['quality_failures']=[x['id'] for x in arm['cases'] if x['quality_status']=='failed']
                        arm['status']='completed_with_quality_failures' if arm['quality_failures'] else 'passed'
                        metric=arm['throughput']['metrics']
                        print(json.dumps({'event':'screen_result','profile':name,'quality_failures':arm['quality_failures'],'eligible':arm['throughput']['eligible_for_speed_comparison'],'c8_common_tps':metric['common_window']['aggregate_output_tokens_per_second'],'c8_wall_tps':metric['whole_request_output_tokens_per_second'],'solo_wall_ms':arm['solo']['wall_ms']}),flush=True)
                finally:
                    stop.set();sampler.join(timeout=15);arm['finished_at']=B.utc()
                    (cli.run_dir/(name+'-report.json')).write_text(json.dumps(arm,indent=2)+'\n')
                guard.check(force=True,before_start=True)
            report['status']='completed'
        except BaseException as error:
            report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error));raise
        finally:
            report['finished_at']=B.utc()
            (cli.run_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    def interrupted(signum,_frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
    main()
