#!/usr/bin/env python3
"""Stage and verify official stock Strata packages without running the engine."""
from __future__ import annotations
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
RELEASE = Path('/home/USER/Documents/strata-lsa-resume-20261103/upstream-review/latest-release.json')
UPSTREAM = Path('/tmp/strata-release-patch-check-nslcv1iq/Strata-06698c674673008395ba6e971580538d6b1ea6e0')
TRUSTED_KEYS = Path('/home/USER/Documents/strata-lsa-resume-20261103/strata-src/strata_control/update_keys.json')
NAMES = ('strata-feed.json', 'strata-feed.json.sig', 'SHA256SUMS.txt',
         'strata-engine-0.1.56-linux-x86_64-cuda-cuda13.tar.gz',
         'strata-control-0.1.56-linux-x86_64.tar.gz')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def fetch(asset):
    path = ROOT / 'assets' / asset['name']
    expected = asset['digest'].removeprefix('sha256:')
    if path.exists():
        if path.stat().st_size != asset['size'] or sha(path) != expected:
            raise RuntimeError('Existing package differs from the release: ' + path.name)
        return path
    partial = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        try:
            start = partial.stat().st_size if partial.exists() else 0
            headers = {'User-Agent': 'Strata-isolated-qualification'}
            if start:
                headers['Range'] = 'bytes=' + str(start) + '-'
            request = urllib.request.Request(asset['browser_download_url'], headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                append = start > 0 and response.status == 206
                with partial.open('ab' if append else 'wb') as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            if partial.stat().st_size != asset['size'] or sha(partial) != expected:
                raise RuntimeError('Downloaded package hash/size differs: ' + path.name)
            os.replace(partial, path)
            print(json.dumps({'event': 'download_verified', 'name': path.name, 'bytes': path.stat().st_size, 'sha256': expected}), flush=True)
            return path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)

def extract(archive, destination, expected_size):
    if destination.exists():
        return
    temporary = destination.with_name(destination.name + '.extracting')
    temporary.mkdir(exist_ok=False)
    try:
        with tarfile.open(archive) as source:
            total = sum(member.size for member in source.getmembers() if member.isfile())
            if total > expected_size:
                raise RuntimeError('Expanded package exceeds signed installed_bytes')
            source.extractall(temporary, filter='data')
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise

def main():
    release = json.loads(RELEASE.read_text())
    if release.get('tag_name') != 'v0.2.10':
        raise RuntimeError('Cached release differs from this qualification pin')
    by_name = {item['name']: item for item in release['assets']}
    assets = [by_name[name] for name in NAMES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, assets))
    sys.path.insert(0, str(UPSTREAM))
    from strata_control import signing, feed, engine_component
    signing.BUILTIN_KEYS = TRUSTED_KEYS
    trusted = signing.load_trusted_keys(environ={})
    verified_feed = feed.verify_and_parse((ROOT/'assets/strata-feed.json').read_bytes(),
                                        (ROOT/'assets/strata-feed.json.sig').read_bytes(),
                                        source=by_name['strata-feed.json']['browser_download_url'],
                                        trusted=trusted, channel='stable')
    engine_package = next(item for item in verified_feed.engine if item.version == '0.1.56'
                          and item.platform == 'linux-x86_64' and item.backend == 'cuda' and item.variant == 'cuda13')
    control_package = next(item for item in verified_feed.control if item.version == '0.1.56' and item.platform == 'linux-x86_64')
    for package, name in ((engine_package, NAMES[-2]), (control_package, NAMES[-1])):
        asset = by_name[name]
        if package.sha256 != asset['digest'].removeprefix('sha256:') or package.size != asset['size']:
            raise RuntimeError('Signed feed and GitHub release disagree for ' + name)
    extract(ROOT/'assets'/NAMES[-2], ROOT/'official-engine-0.1.56', engine_package.installed_bytes)
    extract(ROOT/'assets'/NAMES[-1], ROOT/'official-control-0.1.56', control_package.installed_bytes)
    manifests = list((ROOT/'official-engine-0.1.56').rglob('engine-component.json'))
    if len(manifests) != 1:
        raise RuntimeError('Expected exactly one official engine manifest')
    manifest_path = manifests[0]
    if sha(manifest_path) != engine_package.manifest_sha256:
        raise RuntimeError('Engine manifest differs from signed feed')
    component = engine_component.read_manifest(manifest_path.parent)
    signing.verify_bytes(trusted, engine_component.signing_payload(component.raw), component.signature)
    checked = []
    for relative, digest, size in component.files:
        path = component.root / relative
        if path.stat().st_size != size or sha(path) != digest:
            raise RuntimeError('Component file differs: ' + relative)
        checked.append({'path': str(path), 'bytes': size, 'sha256': digest})
    report = {
        'status': 'verified_not_executed', 'recorded_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'release': release['html_url'], 'release_published_at': release['published_at'],
        'source_pin': '06698c674673008395ba6e971580538d6b1ea6e0',
        'trusted_key_file': str(TRUSTED_KEYS), 'trusted_key_file_sha256': sha(TRUSTED_KEYS),
        'feed': {'key_id': verified_feed.key_id, 'sequence': verified_feed.sequence, 'channel': verified_feed.channel,
                 'generated_at': verified_feed.generated_at, 'expires_at': verified_feed.expires_at,
                 'signature_verified': True},
        'assets': [{'name': a['name'], 'bytes': a['size'], 'sha256': a['digest'].removeprefix('sha256:'),
                    'url': a['browser_download_url']} for a in assets],
        'engine': {'root': str(component.root), 'entrypoint': str(component.root/component.entrypoint),
                   'version': component.version, 'backend': component.backend, 'variant': component.variant,
                   'archs': list(component.archs), 'min_driver': component.min_driver, 'build_id': component.build_id,
                   'manifest_sha256': sha(manifest_path), 'signature_verified': True, 'files': checked},
        'control': {'root': str(ROOT/'official-control-0.1.56'), 'signed_archive_hash_verified': True,
                    'entry': control_package.entry},
        'gpu_work_started': False, 'models_started': False, 'production_files_changed': False,
        'lsa_enabled': False,
    }
    (ROOT/'receipts/official-bundle-verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'engine': report['engine']['entrypoint'], 'control': report['control']['root'],
                      'feed_key_id': verified_feed.key_id, 'files_verified': len(checked)}), flush=True)

if __name__ == '__main__':
    main()
