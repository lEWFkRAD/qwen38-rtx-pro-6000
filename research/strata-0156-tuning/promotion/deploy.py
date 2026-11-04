#!/usr/bin/env python3
"""One-shot approved 0.1.56 promotion; owned HOLD stays in place until separate resume."""
import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile

PROJECT = Path('/home/USER/Documents/strata-0156-tuning-20261103')
SOURCE = PROJECT / 'official-engine-0.1.56'
TARGET = Path('/opt/onyx-strata-0156-06698c674673-cuda13')
OWNER = '<maintenance-owner>'
ROOT = Path('/var/lib/onyx-inference-recovery')
SERVER = Path('/etc/onyx-inference/strata-server.json')
CONTRACT = Path('/etc/onyx-inference/strata.json')
PERSONAL = Path('/etc/onyx-personal/config.json')
EXPECTED = {
    SERVER: 'a73d0dc2ddbf644d1d9f7520d639499e73c7c0f71d38e3d502a05d540464d4a9',
    CONTRACT: '998954a2bf7f26634dead0950d4b52bb16041d6a7b4d609da7661bb38a14c911',
    PERSONAL: '7b244bb7214423dfbd30a604a4deb976e6bab365c65f3780f221575ccf8fa2e6',
    Path('/etc/onyx-inference/strata-server.shared-settings.json'): '376f90ea31c61c38d951df42b85fc8bf892c92984de96060b0f2777403724416',
}
PROFILE = PROJECT / 'profiles/fused_prefill.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def durable(path, data, mode=0o600):
    fd, name = tempfile.mkstemp(prefix='.strata0156-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            os.fchmod(stream.fileno(), mode)
            os.fchown(stream.fileno(), 0, 0)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        d = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(d)
        finally:
            os.close(d)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    assert os.geteuid() == 0, 'root required'
    os.umask(0o077)
    helper = Path('/usr/local/libexec/onyx-inference-recovery')
    maintenance = load_module('promotion_maintenance', helper / 'strata-maintenance.py')
    ctl = maintenance.load_controller()
    stamp = dt.datetime.now(dt.timezone.utc)
    at = stamp.isoformat()
    backup = ROOT / 'backups' / ('strata-0156-promote-' + stamp.strftime('%Y%m%dT%H%M%SZ'))
    with contextlib.ExitStack() as stack:
        for name in ctl.LOCKS:
            fd = stack.enter_context(open(name, 'a'))
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hold = maintenance.require_owner(OWNER)
        old = ctl.contract.load_config()
        ctl.contract.verify_static(old)
        maintenance.inactive_personal(ctl)
        assert maintenance.fully_stopped(ctl, old), 'production must remain stopped'
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        assert not apps, 'GPU has another owner'
        for p, expected in EXPECTED.items():
            assert sha(p) == expected, 'configuration changed: ' + str(p)
        profile_bytes = PROFILE.read_bytes()
        profile_sha = hashlib.sha256(profile_bytes).hexdigest()
        assert profile_sha == 'ba71b5414df050787bab60aff7b0de0f2a4acf198e421d905a19d44b7915d761'
        profile = json.loads(profile_bytes)
        component_bytes = (SOURCE / 'engine-component.json').read_bytes()
        component_sha = hashlib.sha256(component_bytes).hexdigest()
        assert component_sha == '02497065b6118a4b7639f834571b5ac5fe9591a56c57f687617127eca5ed2989'
        component = json.loads(component_bytes)
        assert component['version'] == '0.1.56' and component['build_id'] == '06698c674673+cuda13.0'
        assert not TARGET.exists(), 'immutable install path already exists; inspect before retry'
        names = [entry['path'] for entry in component['files']] + ['engine-component.json']
        expected_payloads = {entry['path']: entry['sha256'] for entry in component['files']}
        expected_payloads['engine-component.json'] = component_sha
        for entry in component['files']:
            assert Path(entry['path']).name == entry['path']
            p = SOURCE / entry['path']
            assert p.is_file() and not p.is_symlink() and p.stat().st_size == entry['bytes']
            assert sha(p) == entry['sha256'], 'signed payload mismatch: ' + str(p)
        backup.mkdir(mode=0o700)
        originals = {p: p.read_bytes() for p in EXPECTED}
        modes = {p: stat.S_IMODE(p.stat().st_mode) for p in EXPECTED}
        for p, data in originals.items():
            durable(backup / ('personal-config.json' if p == PERSONAL else p.name), data)
        # Forensic records only: retry history, HOLD and ready files are never restored.
        for p in (ROOT / 'HOLD', ROOT / 'state.json', ROOT / 'status.json', Path(old['ready_path']),
                  Path('/etc/systemd/system/qwen38-flash-next-strata.service')):
            if p.exists():
                durable(backup / ('before-' + p.name), p.read_bytes())
        evidence_names = [
            'FINAL-SELECTION.json', 'NATIVE-CONFIRMATION.json', 'final_evidence_audit.json',
            'receipts/final-heldout-r1/report.json', 'receipts/api-throughput-final-r3/report.json',
            'receipts/api-features-final-r1/report.json', 'receipts/mixed-final-r2/report.json',
            'receipts/official-bundle-verification.json', 'receipts/independent-openssl-verification.json',
            'profiles/fused_prefill.json', 'RESULTS.md',
        ]
        evidence_bytes = {name: (PROJECT / name).read_bytes() for name in evidence_names}
        evidence_bytes['profiles/fused_prefill.json'] = profile_bytes
        evidence = {str(PROJECT / name): hashlib.sha256(data).hexdigest() for name, data in evidence_bytes.items()}
        for i, name in enumerate(evidence_names):
            durable(backup / ('evidence-%02d-' % i + Path(name).name), evidence_bytes[name])
        qualification = {
            'version': 1, 'at': at, 'authorization': 'Jeff: Promote',
            'status': 'offline-qualified; managed production startup and live canaries required separately',
            'release': '0.1.56', 'source_commit': '06698c674673008395ba6e971580538d6b1ea6e0',
            'profile_sha256': profile_sha, 'profile_env': profile['env'],
            'context_per_slot': 131072, 'slots': 8, 'kv': 'fully resident ordinary INT8',
            'resident_experts': 24576, 'evidence_sha256': evidence,
            'published_evidence_commit': '47c9b33050503ee9bba6fec02a0ac285b423ff4a',
            'quality_limits': {'shared_failed_arithmetic_cases_per_native_arm': 2,
                               'general_quality_equivalence_established': False},
            'LSA_used': False, 'model_weight_training': False,
            'Sunshine_instruction': 'remain stopped; preserve existing enablement',
        }
        qualified = backup / 'offline-qualification.json'
        durable(qualified, encoded(qualification))
        staging = Path(tempfile.mkdtemp(prefix='.onyx-strata-0156-stage-', dir='/opt'))
        try:
            os.chmod(staging, 0o755)
            for name in names:
                p = staging / name
                if name == 'engine-component.json':
                    p.write_bytes(component_bytes)
                else:
                    shutil.copyfile(SOURCE / name, p)
                os.chown(p, 0, 0)
                os.chmod(p, 0o755 if name in ('strata', 'strata-vision', 'strata-split-2-sm120', 'strata-split-2-sm86') else 0o644)
                assert sha(p) == expected_payloads[name], 'staged signed payload mismatch: ' + name
                with p.open('rb') as f:
                    os.fsync(f.fileno())
            os.rename(staging, TARGET)
            parent = os.open(TARGET.parent, os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        bundle = {str(TARGET / name): expected_payloads[name] for name in names}
        for filename, expected in bundle.items():
            assert sha(filename) == expected, 'installed signed payload mismatch: ' + filename
        stage_receipt = backup / 'stage-receipt.json'
        durable(stage_receipt, encoded({'at': at, 'version': '0.1.56', 'root': str(TARGET),
                                      'files_sha256': bundle, 'source': str(SOURCE),
                                      'signature_verification': evidence[str(PROJECT / 'receipts/independent-openssl-verification.json')]}))
        server = json.loads(originals[SERVER])
        server['exe'] = str(TARGET / 'strata')
        server['lib_dirs'] = [str(TARGET)]
        server['vision']['exe'] = str(TARGET / 'strata-vision')
        assert not profile['arg_overrides']
        assert not any('LSA' in key for key in server.get('env', {})), 'unexpected LSA environment'
        server.setdefault('env', {}).update(profile['env'])
        server_bytes = encoded(server)
        server_sha = hashlib.sha256(server_bytes).hexdigest()
        contract = json.loads(originals[CONTRACT])
        assert contract['engine_argv'][1:] == ['--serve', *server['args'], '--batch', '8']
        contract['engine_argv'][0] = server['exe']
        contract['hashes'][str(SERVER)] = server_sha
        contract['hashes'].update(bundle)
        contract['hashes'][str(qualified)] = sha(qualified)
        contract['hashes'][str(stage_receipt)] = sha(stage_receipt)
        contract['engine_package'] = {
            'version': '0.1.56', 'source_commit': qualification['source_commit'],
            'build_id': component['build_id'], 'cuda': '13.0', 'root': str(TARGET),
            'manifest_sha256': bundle[str(TARGET / 'engine-component.json')],
            'stage_receipt': str(stage_receipt), 'stage_receipt_sha256': sha(stage_receipt),
            'trial_receipt': str(qualified),
            'qualification': 'offline validated stock fused_prefill profile; explicit promotion; managed readiness required',
        }
        contract['runtime_version'] = 'Python runtime 0.1.42 with existing compatibility patch; qualified official engine and vision 0.1.56 CUDA 13.0; eight sessions'
        contract['approved_by'] = 'Jeff: Promote (2026-11-03 chat; execution ' + at + ')'
        contract['qualification_receipt'] = str(qualified)
        prior_trial = contract.pop('active_tuning_trial', None)
        if prior_trial:
            contract.setdefault('tuning_history', []).append({'at': at, 'previous_trial': prior_trial,
                'status': 'superseded by fully qualified 0.1.56 fused_prefill profile; prefill32768 retained'})
        contract['active_tuning_profile'] = {'name': 'fused_prefill', 'selected_at': at,
            'profile_sha256': profile_sha, 'qualification_receipt': str(qualified),
            'engine_environment_delta': profile['env'], 'prefill_chunk': 32768,
            'status': 'approved production profile; live readiness published separately'}
        contract.setdefault('reseal_history', []).append({'at': at, 'reason': 'Jeff explicitly promoted qualified stock Strata0.1.56 fused_prefill',
            'previous_contract_sha256': EXPECTED[CONTRACT], 'previous_server_sha256': EXPECTED[SERVER],
            'backup': str(backup), 'wrapper_source_commit_unchanged': contract['source_commit']})
        contract_bytes = encoded(contract)
        contract_sha = hashlib.sha256(contract_bytes).hexdigest()
        personal = json.loads(originals[PERSONAL])
        assert personal['baseline_sha256'][str(CONTRACT)] == EXPECTED[CONTRACT]
        personal['baseline_sha256'][str(CONTRACT)] = contract_sha
        if str(SERVER) in personal['baseline_sha256']:
            personal['baseline_sha256'][str(SERVER)] = server_sha
        q = personal.setdefault('qualification', {})
        q['primary_contract'] = contract_sha
        q['primary_promotion'] = str(backup)
        q['primary_runtime'] = 'Qualified official engine/vision0.1.56 CUDA13.0; Python wrapper0.1.42 retained; eight slots; prefill32768; spec6'
        q['engine_update_0156'] = {'at': at, 'backup': str(backup), 'authorization': 'Jeff: Promote',
                                   'personal_model_changed': False, 'gpu_swap_tested': False}
        new = {SERVER: server_bytes, CONTRACT: contract_bytes, PERSONAL: encoded(personal)}
        # Recheck immediately before the three guarded replacements.
        assert maintenance.require_owner(OWNER) == hold
        assert maintenance.fully_stopped(ctl, old)
        for p, expected in EXPECTED.items():
            assert sha(p) == expected, 'concurrent edit detected: ' + str(p)
        changed = []
        try:
            for p, data in new.items():
                changed.append(p)
                durable(p, data, modes[p])
            ctl.contract.verify_static(ctl.contract.load_config())
            for filename, expected in personal['baseline_sha256'].items():
                assert sha(filename) == expected, 'personal baseline pin mismatch: ' + filename
            assert maintenance.require_owner(OWNER) == hold
            assert maintenance.fully_stopped(ctl, ctl.contract.load_config())
        except BaseException:
            for p in reversed(changed):
                current = p.read_bytes()
                if current == new[p]:
                    durable(p, originals[p], modes[p])
                elif current != originals[p]:
                    raise RuntimeError('Unexpected concurrent edit during rollback; HOLD retained; preserved ' + str(p))
            ctl.contract.verify_static(ctl.contract.load_config())
            raise
        result = {'version': 1, 'at': at, 'status': 'deployed_static_verified_owned_HOLD_retained',
                  'backup': str(backup), 'root': str(TARGET),
                  'new_config_sha256': {str(p): sha(p) for p in new},
                  'old_config_sha256': {str(p): h for p, h in EXPECTED.items()},
                  'qualification_receipt': str(qualified), 'stage_receipt': str(stage_receipt),
                  'resume_owner': OWNER, 'state_history_modified': False, 'Sunshine_modified': False}
        durable(backup / 'deployment-result.json', encoded(result))
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
