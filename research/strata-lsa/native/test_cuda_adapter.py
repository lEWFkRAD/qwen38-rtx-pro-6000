#!/usr/bin/env python3
"""Bounded native LSA bridge qualification. Default is a plan; --prepare is CPU only.

The bridge is not called by Strata's serving engine. Ward clustering remains the
pinned upstream implementation. A passing result is component parity only.
"""
from __future__ import annotations
import argparse
import contextlib
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
RESEARCH = Path('/path/to/lsa-prep-20261103')
MAPROOT = RESEARCH / 'flashnext-maps'
PYTHON = RESEARCH / '.venv/bin/python'
LIBRARY = ROOT / 'build-cuda/libstrata_lsa_cuda.so'
NATIVE_MAPS = ROOT / 'receipts/flashnext-learned-native-v1.bin'
MAINTENANCE = '/usr/local/libexec/onyx-inference-recovery/strata-maintenance.py'
SERVICE = 'qwen38-flash-next-strata.service'
EXPECTED_OWNER = 'example-lsa-maintenance-owner'
MAP_SHA = '4d8a606e0ddd33858cb662e7f82f723dd54ca96776f3c383af28b30935a85d02'
LAYERS = tuple(range(3, 48, 4))
WIDTH, TOP_P = 1540, 385


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def owned_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeError('Output must stay in the Documents experiment directory')
    return path


def write_exclusive(path, value):
    path = owned_output(path)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def command(argv, timeout=10):
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f'Read failed ({completed.returncode}): {argv[0]}: {completed.stderr[:400]}')
    return completed.stdout


def service_state():
    raw = command(['systemctl', 'show', SERVICE, '--property=LoadState', '--property=ActiveState'])
    fields = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    if fields.get('LoadState') != 'loaded':
        raise RuntimeError('Production service state cannot be established')
    return fields.get('ActiveState')


def maintenance_state(owner, timeout=15):
    if owner != EXPECTED_OWNER:
        raise RuntimeError('Unexpected maintenance owner')
    status = json.loads(command(['sudo', '-n', 'python3', MAINTENANCE, 'status'], timeout=timeout))
    hold = status.get('hold')
    if status.get('status') != 'maintenance' or not isinstance(hold, dict) or hold.get('owner') != owner:
        raise RuntimeError('Owned maintenance HOLD is no longer established')
    if service_state() != 'inactive':
        raise RuntimeError('Production became active')
    return dict(status=status['status'], owner=hold['owner'], service='inactive')


def gpu_snapshot():
    import csv
    row = list(csv.reader(command(['nvidia-smi', '--id=0',
        '--query-gpu=uuid,name,compute_cap,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits']).splitlines(), skipinitialspace=True))
    if len(row) != 1 or len(row[0]) != 5:
        raise RuntimeError('Cannot establish one GPU identity')
    ident, name, cap, free, util = row[0]
    return dict(uuid=ident, name=name, capability=cap, free_mib=int(free), utilization=int(util))


def load_capture(path):
    """Read raw INT8 evidence without writing/repacking the original artifact."""
    import numpy as np
    import torch
    raw = Path(path).read_bytes()
    if len(raw) < 144:
        raise ValueError('Truncated capture header')
    h = struct.unpack_from('<18q', raw)
    magic, version, layer, ordinal, p0, tokens, keys, nq = h[:8]
    if not (magic == 0x4c53414d41503031 and version == 1 and layer in LAYERS
            and ordinal == (layer - 3) // 4 and p0 == 0 and keys == tokens
            and h[8:12] == (24, 2, 256, 64) and h[14:] == (4, 32, 0, 0)
            and h[12] == 4 and 1024 <= tokens <= 8192 and 1 <= nq <= 258):
        raise ValueError('Capture identity/geometry mismatch')
    cap, offset = h[13], 144
    def take(dtype, shape):
        nonlocal offset
        count = int(np.prod(shape))
        dt = np.dtype(dtype)
        end = offset + count * dt.itemsize
        if end > len(raw):
            raise ValueError('Truncated capture body')
        value = torch.from_numpy(np.frombuffer(raw, dtype=dt, count=count, offset=offset).reshape(shape).copy())
        offset = end
        return value
    positions = take('<i4', (nq,))
    q = take('<f4', (nq, 24, 256))
    actual = take('<f4', (nq, 24, 256))
    ids = take('<i4', (nq, cap))
    steps = take('<i4', (nq, 4))
    kc, vc = take('i1', (tokens, 2, 256)), take('i1', (tokens, 2, 256))
    ks, vs = take('<f2', (tokens, 2, 4)), take('<f2', (tokens, 2, 4))
    if offset != len(raw) or not torch.equal(steps[:, 0], positions) or not torch.equal(steps[:, 1], positions + 1):
        raise ValueError('Capture length/step mismatch')
    if not bool(((positions >= 0) & (positions < tokens)).all()) or not bool((positions[1:] > positions[:-1]).all()):
        raise ValueError('Capture positions invalid')
    if not all(bool(torch.isfinite(t).all()) for t in (q, actual, ks, vs)):
        raise ValueError('Capture contains nonfinite inputs')
    if not bool((ks >= 0).all()) or not bool((vs >= 0).all()):
        raise ValueError('Capture scales invalid')
    k = kc.float() * ks.float().repeat_interleave(64, -1)
    v = vc.float() * vs.float().repeat_interleave(64, -1)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), layer=layer, tokens=tokens,
                positions=positions, query=q, codes_k=kc, codes_v=vc, scales_k=ks, scales_v=vs,
                key=k, value=v)


def cpu_expand(picks, members, position):
    if position < 0 or position >= 131072:
        return []
    window = max(0, position + 1 - 512)
    result = []
    for bucket in picks:
        if bucket < 0:
            continue
        for item in members[bucket]:
            if 4 <= item < window and len(result) < 1024:
                result.append(item)
    result.extend(range(window, position + 1))
    result.extend(range(min(4, window, position + 1)))
    return result


def prepare():
    """CPU-only input/ABI preparation; does not import or initialize CUDA."""
    import torch
    torch.set_num_threads(2)
    path = MAPROOT / 'maps/pilot-v1/flashnext-learned-384x64.pt'
    if sha(path) != MAP_SHA:
        raise RuntimeError('Immutable learned map hash changed')
    blob = torch.load(path, map_location='cpu', weights_only=True)
    expected = {f'{layer}.{kind}' for layer in LAYERS for kind in ('W_Q_idx', 'W_K_idx')}
    if set(blob) != expected:
        raise RuntimeError('Unexpected map keys')
    for value in blob.values():
        if tuple(value.shape) != (384, 64) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise RuntimeError('Invalid map values')
    captures = []
    for layer in LAYERS:
        data = load_capture(MAPROOT / 'captures/test_code' / f'layer-{layer}-p0.lsacap')
        captures.append(dict(layer=layer, tokens=data['tokens'], queries=len(data['positions']), sha256=data['sha256']))
    # Boundary reference checks do not mirror native CUDA code: the independent
    # long-range/window/sink set is compared with the straightforward expansion.
    members = [[-1] * 16 for _ in range(12 * 64)]
    for b in range(len(members)):
        members[b][:8] = range(b * 8, b * 8 + 8)
    for position in (0, 3, 4, 511, 512, 513, 1023, 1024, 1535, 1536, 6143):
        visible = min(position // 512 * 64, len(members))
        result = cpu_expand(list(range(min(TOP_P, visible))), members, position)
        assert len(result) <= WIDTH and len(result) == len(set(result))
        assert all(0 <= item <= position for item in result)
        assert set(range(max(0, position + 1 - 512), position + 1)).issubset(result)
        assert set(range(min(4, position + 1))).issubset(result)
    assert cpu_expand([0], members, -1) == []
    assert cpu_expand([0], members, 131072) == []
    return dict(status='cpu_prepared', gpu_executed=False, map_sha256=MAP_SHA,
                captures=captures, boundary_reference_checks=13,
                library_present=LIBRARY.is_file(), script_sha256=sha(__file__))


class Bridge:
    def __init__(self, stream, library):
        self.lib = C.CDLL(str(library))
        self.stream = C.c_void_p(stream.cuda_stream)
        if not self.stream.value:
            raise RuntimeError('Native bridge requires an explicit nondefault stream')
        ptr, integer = C.c_void_p, C.c_int
        signatures = {
            'lsa_project_keys': [ptr, ptr, ptr, integer, integer, integer, integer, ptr, ptr, ptr],
            'lsa_project_key_steps': [ptr, ptr, ptr, integer, integer, ptr, integer, ptr, ptr, ptr],
            'lsa_project_queries': [ptr, integer, ptr, ptr, ptr],
            'lsa_select_temp_bytes': [integer, integer, C.POINTER(C.c_size_t)],
            'lsa_select': [ptr, ptr, ptr, ptr, integer, integer, ptr, ptr, ptr, ptr, C.c_size_t, ptr, ptr, ptr, ptr],
            'lsa_attention': [ptr, ptr, ptr, ptr, ptr, ptr, integer, integer, ptr, ptr, ptr, ptr, ptr],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, C.c_int
        self.lib.lsa_maps_open.argtypes = [C.c_char_p, C.c_void_p, C.c_size_t]
        self.lib.lsa_maps_open.restype = C.c_void_p
        for name in ('lsa_maps_query', 'lsa_maps_key'):
            getattr(self.lib, name).argtypes = [C.c_void_p, C.c_int]
            getattr(self.lib, name).restype = C.c_void_p
        self.lib.lsa_maps_temperature.argtypes = [C.c_void_p, C.c_int]
        self.lib.lsa_maps_temperature.restype = C.c_float
        self.lib.lsa_maps_close.argtypes = [C.c_void_p]
        self.lib.lsa_maps_close.restype = None
        self.lib.lsa_attention_scratch_floats.argtypes = []
        self.lib.lsa_attention_scratch_floats.restype = C.c_size_t

    @staticmethod
    def p(tensor):
        if isinstance(tensor, int):
            return C.c_void_p(tensor)
        if isinstance(tensor, C.c_void_p):
            return tensor
        return C.c_void_p(tensor.data_ptr())

    def call(self, name, *args):
        rc = getattr(self.lib, name)(*args)
        if rc:
            raise RuntimeError(f'{name} returned CUDA error {rc}')

    def project_keys(self, pools, w, out):
        self.call('lsa_project_keys', self.p(pools['k']), self.p(pools['ks']), self.p(pools['table']),
                  pools['pages'], 4, 0, out.shape[0], self.p(w), self.p(out), self.stream)

    def project_key_steps(self, pools, positions, w, out):
        self.call('lsa_project_key_steps', self.p(pools['k']), self.p(pools['ks']), self.p(pools['table']),
                  pools['pages'], 4, self.p(positions), len(positions), self.p(w), self.p(out), self.stream)

    def project_queries(self, q, w, out):
        self.call('lsa_project_queries', self.p(q), len(q), self.p(w), self.p(out), self.stream)

    def buffers(self, queries, buckets):
        import torch
        needed = C.c_size_t()
        self.call('lsa_select_temp_bytes', queries, buckets, C.byref(needed))
        return dict(keys_in=torch.empty((queries, buckets), dtype=torch.int64, device='cuda'),
                    keys_out=torch.empty((queries, buckets), dtype=torch.int64, device='cuda'),
                    offsets=torch.empty(queries + 1, dtype=torch.int32, device='cuda'),
                    temporary=torch.empty(max(1, needed.value), dtype=torch.uint8, device='cuda'),
                    selected=torch.empty((queries, TOP_P), dtype=torch.int32, device='cuda'),
                    cells=torch.empty((queries, WIDTH), dtype=torch.int32, device='cuda'),
                    steps=torch.empty((queries, 4), dtype=torch.int32, device='cuda'),
                    scratch=torch.empty((queries, self.lib.lsa_attention_scratch_floats()), device='cuda'),
                    attention=torch.empty((queries, 24, 256), device='cuda'))

    def select(self, q, centroids, members, positions, b):
        self.call('lsa_select', self.p(q), self.p(centroids), self.p(members), self.p(positions), len(q),
                  len(centroids), self.p(b['keys_in']), self.p(b['keys_out']), self.p(b['offsets']),
                  self.p(b['temporary']), b['temporary'].numel(), self.p(b['selected']), self.p(b['cells']),
                  self.p(b['steps']), self.stream)

    def attention(self, q, pools, b):
        self.call('lsa_attention', self.p(q), self.p(pools['k']), self.p(pools['v']), self.p(pools['ks']),
                  self.p(pools['vs']), self.p(pools['table']), 4, len(q), self.p(b['cells']), self.p(b['steps']),
                  self.p(b['scratch']), self.p(b['attention']), self.stream)


def paged_pools(data, seed=1977):
    import torch
    tokens = data['tokens']
    assert tokens % 4 == 0
    pages = tokens // 4
    permutation = torch.randperm(pages, generator=torch.Generator().manual_seed(seed))
    result = dict(pages=pages, table=permutation.cuda().int().contiguous())
    for name, source in [('k', 'codes_k'), ('v', 'codes_v'), ('ks', 'scales_k'), ('vs', 'scales_v')]:
        value = data[source].reshape(pages, 4, 2, -1).permute(0, 2, 1, 3).contiguous()
        packed = torch.empty_like(value)
        packed[permutation] = value
        result[name] = packed.cuda().contiguous()
    return result


def cluster_index(keys):
    import torch
    import torch.nn.functional as F
    from lsa.core.kernels.ward import cluster_ward
    from lsa.core.kernels.triton.ward import bucket_members_from_labels
    blocks = len(keys) // 512
    closed = keys[:blocks * 512].reshape(blocks, 512, 64)
    pooled = closed.reshape(blocks, 128, 4, 64).sum(2)
    sizes = torch.full((blocks, 128), 4., device='cuda')
    labels, reached = cluster_ward(pooled, sizes, 64, 16, precision='ieee', kernels='triton')
    if not bool(reached.all()):
        raise RuntimeError('Upstream Ward failed to complete a real-input block')
    labels = labels.repeat_interleave(4, 1)
    sums = torch.zeros((blocks, 64, 64), device='cuda')
    sums.scatter_add_(1, labels[..., None].expand_as(closed), closed)
    centroids = F.normalize(sums, dim=-1).reshape(-1, 64).bfloat16().float().contiguous()
    members = bucket_members_from_labels(labels, 64, 16)
    members = torch.where(members >= 0, members + (torch.arange(blocks, device='cuda') * 512)[:, None, None], -1)
    members = members.reshape(-1, 16).int().contiguous()
    flat = members[members >= 0]
    assert torch.equal(flat.sort().values, torch.arange(blocks * 512, device='cuda'))
    return centroids, members


def selection_parity(bridge, iq, centroids, members, positions, buffers):
    import torch
    from lsa.core.kernels.scan_select import ScanSelect, dense_keys
    from lsa.core.kernels.expansion import expand_in_order
    from lsa.core.kernels.triton.keys import unpack_scores
    bridge.select(iq, centroids, members, positions, buffers)
    visible = (positions // 512 * 64).clamp(0, len(centroids)).int().contiguous()
    oracle = ScanSelect(len(iq), TOP_P, device='cuda', kernels='triton')
    picks = oracle(iq.bfloat16(), centroids.bfloat16(), visible).clone()
    expanded, lengths = expand_in_order(picks, members, positions, 512, 4, 1024, kernels='triton')
    native_picks, native_cells, native_steps = (buffers[key].cpu() for key in ('selected', 'cells', 'steps'))
    expected_picks, expected_cells, expected_lengths = picks.cpu(), expanded.cpu(), lengths.cpu()
    if not torch.equal(native_picks, expected_picks):
        mismatch = (native_picks != expected_picks).nonzero().tolist()
        raise AssertionError(f'Native bucket ranking differs from upstream at {mismatch[:12]}')
    assert torch.equal(native_cells, expected_cells), 'Native expansion differs from upstream ordered expansion'
    assert torch.equal(native_steps[:, 3], expected_lengths), 'Native selected widths differ'
    assert torch.equal(native_steps[:, 0], positions.cpu())
    assert torch.equal(native_steps[:, 1], positions.cpu() + 1)
    reference_keys = dense_keys(iq.bfloat16(), centroids.bfloat16(), visible, kernels='triton')
    actual_scores, wanted_scores = unpack_scores(buffers['keys_in']).cpu(), unpack_scores(reference_keys).cpu()
    scores = []
    for row, count in enumerate(visible.cpu().tolist()):
        if count:
            scores.append(float((actual_scores[row, :count] - wanted_scores[row, :count]).abs().max()))
        held = native_cells[row, :int(expected_lengths[row])]
        assert len(held) == held.unique().numel()
        assert bool(((held >= 0) & (held <= positions[row].item())).all())
    max_score_error = max(scores, default=0.)
    assert max_score_error <= 2e-6, f'Centroid scores differ materially: {max_score_error}'
    return dict(bucket_order_exact=True, expansion_exact=True, width_exact=True,
                max_score_abs_error=max_score_error, lengths=expected_lengths.tolist())


def attention_parity(bridge, q, pools, data, buffers):
    import torch
    bridge.attention(q, pools, buffers)
    actual, cells, steps = buffers['attention'].cpu().double(), buffers['cells'].cpu(), buffers['steps'].cpu()
    query = q.cpu().double()
    key, value = data['key'].double(), data['value'].double()
    errors = []
    for row in range(len(q)):
        held = cells[row, :int(steps[row, 3])].long()
        assert len(held) > 0 and len(held.unique()) == len(held)
        assert bool(((held >= 0) & (held < data['tokens'])).all())
        scores = torch.einsum('hgd,nhd->hgn', query[row].reshape(2, 12, 256), key[held]) / 16
        wanted = torch.einsum('hgn,nhd->hgd', scores.softmax(-1), value[held]).reshape(24, 256)
        delta = actual[row] - wanted
        relative_max = float(delta.abs().max()) / max(float(wanted.abs().max()), 1e-12)
        relative_l2 = float(delta.norm()) / max(float(wanted.norm()), 1e-12)
        assert torch.isfinite(actual[row]).all()
        assert relative_max <= 1e-4 and relative_l2 <= 1e-4, (row, relative_max, relative_l2)
        errors.append(dict(relative_max_scale=relative_max, relative_l2=relative_l2))
    return dict(queries=len(q), max_relative_max_scale=max(e['relative_max_scale'] for e in errors),
                max_relative_l2=max(e['relative_l2'] for e in errors),
                oracle='FP64 same selected set, original FP32 INT8-code times FP16-scale values',
                native_reader='unchanged Strata qsa_decode_attn_batch')


def graph_parity(bridge, stream, pools, native_wq, native_wk, q, iq, ik, centroids, members, data):
    import torch
    positions = torch.tensor([511, 512, 513, 1023, 1024, 1535, 1536, data['tokens'] - 1], dtype=torch.int32, device='cuda')
    kout = torch.empty((8, 64), device='cuda')
    qout = torch.empty_like(iq)
    buffers = bridge.buffers(8, len(centroids))
    def operations():
        bridge.project_key_steps(pools, positions, native_wk, kout)
        bridge.project_queries(q, native_wq, qout)
        bridge.select(qout, centroids, members, positions, buffers)
        bridge.attention(q, pools, buffers)
    for _ in range(2):
        operations()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        operations()
    replays = []
    for values in ([511, 512, 513, 1023, 1024, 1535, 1536, data['tokens'] - 1],
                   [0, 31, 1024, 2048, 4095, 4096, data['tokens'] - 2, data['tokens'] - 1]):
        positions.copy_(torch.tensor(values, dtype=torch.int32, device='cuda'))
        graph.replay()
        captured = {name: buffers[name].clone() for name in ('selected', 'cells', 'steps', 'attention')}
        captured_k, captured_q = kout.clone(), qout.clone()
        operations()
        for name in captured:
            assert torch.equal(captured[name], buffers[name]), f'Graph/direct mismatch in {name}'
        assert torch.equal(captured_k, kout) and torch.equal(captured_q, qout)
        assert torch.equal(kout, ik[positions.long()]), 'Device-position key projection changed arithmetic'
        selection = selection_parity(bridge, qout, centroids, members, positions, buffers)
        # Deliberately reuse the full future-populated centroid allocation at
        # boundary positions. Equality to causal upstream selection proves masking.
        replays.append(dict(positions=values, direct_exact=True, future_centroids_masked=True,
                            selection=selection))
    return dict(capture_succeeded=True, replays=replays)


def synthetic_checks(bridge, stream, max_capacity=False):
    import torch
    reports = []
    buckets = 16384 if max_capacity else 768
    members = torch.full((buckets, 16), -1, dtype=torch.int32, device='cuda')
    members[:, :8] = torch.arange(buckets * 8, dtype=torch.int32, device='cuda').reshape(-1, 8)
    positions = torch.tensor([511, 512, 1536, 131071 if max_capacity else 6143], dtype=torch.int32, device='cuda')
    iq = torch.zeros((4, 64), device='cuda')
    centroids = torch.zeros((buckets, 64), device='cuda')
    centroids[:, 0] = 1
    buffers = bridge.buffers(4, buckets)
    reports.append(dict(case='zero_query_equal_scores', result=selection_parity(bridge, iq, centroids, members, positions, buffers)))
    iq[:, 0] = 1
    reports.append(dict(case='one_hot_equal_scores', result=selection_parity(bridge, iq, centroids, members, positions, buffers)))
    iq.zero_()
    centroids[:] = -0.125
    centroids[1::2] = 0.125
    reports.append(dict(case='signed_zero_scores', result=selection_parity(bridge, iq, centroids, members, positions, buffers)))
    # Invalid logical positions are rejected before a page-table access; invalid
    # physical page IDs map to zero projected K. No invalid table reaches attention.
    dummy = dict(tokens=512, codes_k=torch.ones((512, 2, 256), dtype=torch.int8),
                 codes_v=torch.zeros((512, 2, 256), dtype=torch.int8),
                 scales_k=torch.ones((512, 2, 4), dtype=torch.float16),
                 scales_v=torch.ones((512, 2, 4), dtype=torch.float16))
    pools = paged_pools(dummy)
    pools['table'][0], pools['table'][1] = -1, pools['pages']
    bad_positions = torch.tensor([-1, 131072, 0, 4], device='cuda', dtype=torch.int32)
    out, weight = torch.empty((4, 64), device='cuda'), torch.ones((384, 64), device='cuda')
    bridge.project_key_steps(pools, bad_positions, weight, out)
    assert torch.equal(out, torch.zeros_like(out)), 'Invalid key projection did not yield finite zero'
    zeroq = torch.zeros((4, 24, 256), device='cuda')
    bridge.project_queries(zeroq, weight, out)
    assert torch.equal(out, torch.zeros_like(out)), 'Zero Q projection is not zero'
    reports.append(dict(case='zero_and_invalid_projection', passed=True))
    stream.synchronize()
    return reports


def worker(args):
    if os.environ.get('LSA_SUPERVISED_OWNER') != args.owner or args.owner != EXPECTED_OWNER:
        raise RuntimeError('Worker requires the bounded parent supervisor and current owner')
    if service_state() != 'inactive':
        raise RuntimeError('Production must be inactive before worker initialization')
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(4096 * 1048576 / torch.cuda.get_device_properties(0).total_memory, 0)
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError('Expected RTX PRO6000 compute capability12.0')
    sys.path.insert(0, str(RESEARCH / 'upstream-research/src'))
    from lsa.sglang.index.weights import read
    map_path = MAPROOT / 'maps/pilot-v1/flashnext-learned-384x64.pt'
    if sha(map_path) != MAP_SHA:
        raise RuntimeError('Immutable map hash changed')
    maps = read(map_path, 384, 64, LAYERS)
    report = dict(status='running', started_at=utc(), gpu_executed=True,
                  scope='Standalone native CUDA projection/selection/expansion and unchanged Strata attention parity',
                  limitations=['No Strata engine call sites are wired', 'Ward cache construction uses pinned upstream Triton',
                               'No incremental native Ward cache, MTP, conversation cache, model quality or production promotion claim'],
                  map_sha256=MAP_SHA, library_sha256=sha(args.library), native_map_sha256=sha(NATIVE_MAPS), script_sha256=sha(__file__),
                  gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__, cases=[])
    stream = torch.cuda.Stream()
    bridge = Bridge(stream, args.library)
    error_buffer = C.create_string_buffer(1024)
    map_handle = bridge.lib.lsa_maps_open(os.fsencode(NATIVE_MAPS), error_buffer, len(error_buffer))
    if not map_handle:
        raise RuntimeError('Native map load failed: ' + error_buffer.value.decode(errors='replace'))
    temperatures = json.loads((MAPROOT / 'maps/pilot-v1/flashnext-learned-384x64.json').read_text())['tau']
    report['native_map_loader'] = dict(loaded=True, path=str(NATIVE_MAPS), temperatures={})
    try:
        with torch.cuda.stream(stream):
            for layer in LAYERS:
                if service_state() != 'inactive':
                    raise RuntimeError('Production state changed during worker')
                data = load_capture(MAPROOT / 'captures/test_code' / f'layer-{layer}-p0.lsacap')
                eligible = (data['positions'] >= 1536).nonzero().flatten()
                chosen = eligible[torch.linspace(0, len(eligible) - 1, 8).round().long()]
                q = data['query'][chosen].cuda().contiguous()
                positions = data['positions'][chosen].cuda().int().contiguous()
                wq, wk = (value.cuda().contiguous() for value in maps[layer])
                native_wq = bridge.lib.lsa_maps_query(map_handle, layer)
                native_wk = bridge.lib.lsa_maps_key(map_handle, layer)
                tau = float(bridge.lib.lsa_maps_temperature(map_handle, layer))
                assert native_wq and native_wk and tau > 0 and abs(tau - temperatures[str(layer)]) < 1e-6
                report['native_map_loader']['temperatures'][str(layer)] = tau
                pools = paged_pools(data, seed=1977 + layer)
                ik = torch.empty((data['tokens'], 64), device='cuda')
                iq = torch.empty((8, 64), device='cuda')
                bridge.project_keys(pools, native_wk, ik)
                bridge.project_queries(q, native_wq, iq)
                key_input = data['key'][:, :, 64:].reshape(-1, 384).cuda()
                query_input = q.reshape(8, 2, 12, 256)[:, :, :, 64:].sum(2).reshape(8, 384)
                expected_k = F.normalize(key_input @ wk, dim=-1)
                expected_q_fp32 = F.normalize(query_input @ wq, dim=-1)
                expected_q = expected_q_fp32.bfloat16().float()
                torch.testing.assert_close(ik, expected_k, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(iq, expected_q, rtol=0, atol=0.004)
                assert torch.isfinite(ik).all() and torch.isfinite(iq).all()
                projection = dict(key_max_abs_error=float((ik - expected_k).abs().max()),
                                  query_bf16_differing_coordinates=int((iq != expected_q).sum()),
                                  query_bf16_max_abs_difference=float((iq - expected_q).abs().max()),
                                  fp32_oracle='torch FP32 matmul with TF32 disabled, then normalize',
                                  rounding_note='Native serial FP32 dot and cuBLAS reduction may cross BF16 rounding boundaries')
                centroids, members = cluster_index(ik)
                buffers = bridge.buffers(8, len(centroids))
                selection = selection_parity(bridge, iq, centroids, members, positions, buffers)
                attention = attention_parity(bridge, q, pools, data, buffers)
                case = dict(layer=layer, tokens=data['tokens'], query_positions=positions.cpu().tolist(),
                            capture_sha256=data['sha256'], physical_pages_permuted=True,
                            projection=projection, selection=selection, attention=attention)
                if layer in (3, 47):
                    case['graphs'] = graph_parity(bridge, stream, pools, native_wq, native_wk, q, iq, ik, centroids, members, data)
                report['cases'].append(case)
                print(json.dumps(dict(event='layer_passed', layer=layer, projection=projection, attention=attention)), flush=True)
                del data, q, positions, wq, wk, pools, ik, iq, key_input, query_input, expected_k, expected_q_fp32, expected_q, centroids, members, buffers
            report['synthetic'] = synthetic_checks(bridge, stream, args.max_capacity)
            stream.synchronize()
        if service_state() != 'inactive':
            raise RuntimeError('Production changed before final worker check')
        report.update(status='passed', peak_allocated_mib=torch.cuda.max_memory_allocated() / 1048576,
                      peak_reserved_mib=torch.cuda.max_memory_reserved() / 1048576)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        bridge.lib.lsa_maps_close(map_handle)
        report['finished_at'] = utc()
        write_exclusive(args.output, report)


def kill_owned(proc):
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)


def execute(args):
    if args.owner != EXPECTED_OWNER:
        raise RuntimeError('--execute requires the current owned maintenance owner')
    before = maintenance_state(args.owner)
    gpu = gpu_snapshot()
    # This run establishes correctness, not uncontended performance. Sunshine
    # keeps the desktop at about 8% GPU load while no model server is resident.
    # Keep a strict default, and explicitly record a bounded desktop allowance.
    utilization_limit = 20 if args.allow_desktop_load else 5
    if args.allow_desktop_load:
        import csv
        processes = list(csv.DictReader(command(['nvidia-smi', '--query-compute-apps=process_name', '--format=csv']).splitlines()))
        if any(row.get('process_name', '').strip() != '/usr/bin/sunshine' for row in processes):
            raise RuntimeError('Desktop allowance requires Sunshine to be the only compute process')
    if gpu['capability'] != '12.0' or gpu['free_mib'] < 8192 or gpu['utilization'] > utilization_limit:
        raise RuntimeError('GPU window is not idle with required free memory')
    if not args.library.is_file():
        raise RuntimeError('Native shared library is not built')
    run = ROOT / 'receipts' / ('cuda-adapter-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8])
    run.mkdir(exist_ok=False)
    result_path = run / 'result.json'
    receipt = dict(started_at=utc(), status='launching', before=before, gpu=gpu,
                   script_sha256=sha(__file__), library_sha256=sha(args.library), timeout_seconds=args.max_seconds,
                   owner=args.owner, gpu_work_started=False,
                   preflight_utilization_limit=utilization_limit,
                   desktop_load_allowed=args.allow_desktop_load,
                   performance_claim=False)
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0', LSA_SUPERVISED_OWNER=args.owner,
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    env.pop('NVIDIA_VISIBLE_DEVICES', None)
    for key, name in [('TRITON_CACHE_DIR', 'triton'), ('CUDA_CACHE_PATH', 'cuda'),
                      ('TORCH_EXTENSIONS_DIR', 'extensions'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('XDG_CACHE_HOME', 'xdg')]:
        destination = ROOT / 'cache' / name
        destination.mkdir(parents=True, exist_ok=True)
        env[key] = str(destination)
    argv = [str(PYTHON), str(Path(__file__).resolve()), '--worker', '--owner', args.owner, '--output', str(result_path), '--library', str(args.library)]
    if args.max_capacity:
        argv.append('--max-capacity')
    proc = None
    try:
        with (run / 'worker.log').open('x', encoding='utf-8') as log:
            maintenance_state(args.owner)
            proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt.update(gpu_work_started=True, pid=proc.pid, argv=argv)
            deadline = time.monotonic() + args.max_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Native bridge worker exceeded external deadline')
                try:
                    rc = proc.wait(timeout=min(5, remaining))
                    break
                except subprocess.TimeoutExpired:
                    maintenance_state(args.owner, timeout=min(15, max(.1, deadline - time.monotonic())))
            receipt['returncode'] = rc
        receipt['after'] = maintenance_state(args.owner)
        if rc != 0 or not result_path.is_file():
            raise RuntimeError(f'Worker failed with exit {rc}; inspect retained worker.log')
        result = json.loads(result_path.read_text())
        if result.get('status') != 'passed' or result.get('gpu_executed') is not True or len(result.get('cases', [])) != 12:
            raise RuntimeError('Worker result does not establish all12 passing component cases')
        receipt.update(status='component_parity_passed', result_sha256=sha(result_path))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            kill_owned(proc)
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt['finished_at'] = utc()
        write_exclusive(run / 'receipt.json', receipt)
        print(json.dumps(dict(status=receipt['status'], receipt=str(run / 'receipt.json'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--worker', action='store_true')
    parser.add_argument('--owner')
    parser.add_argument('--library', type=Path, default=LIBRARY)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-seconds', type=int, default=900)
    parser.add_argument('--max-capacity', action='store_true', help='Also scan16384 synthetic buckets at131072capacity')
    parser.add_argument('--allow-desktop-load', action='store_true', help='Correctness only: permit <=20% GPU activity when Sunshine is the only compute process')
    args = parser.parse_args()
    if not 60 <= args.max_seconds <= 1800:
        parser.error('--max-seconds must be60..1800')
    if args.worker and args.output is None:
        parser.error('--worker requires --output')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    if args.prepare:
        value = prepare()
        if args.output:
            write_exclusive(args.output, value)
        print(json.dumps(value, indent=2), flush=True)
    elif args.worker:
        worker(args)
    elif args.execute:
        execute(args)
    else:
        print(json.dumps(dict(mode='plan', gpu_executed=False, prepare='--prepare performs CPU-only evidence checks',
                              execute='--execute --owner requires established attended maintenance and starts bounded worker',
                              scope='Native CUDA bridge parity, no engine integration/promotion claim'), indent=2))


if __name__ == '__main__':
    main()
