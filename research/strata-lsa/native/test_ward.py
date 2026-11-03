#!/usr/bin/env python3
"""Native Ward qualification only. Default plan/--prepare are CPU only.

GPU execution must be coordinated through the existing owned-maintenance parent.
The native kernel is spherical mutual-nearest Ward, not greedy agglomeration.
"""
from __future__ import annotations
import argparse
import ctypes as C
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
import test_cuda_adapter as base
ROOT=base.ROOT

class Ward:
    def __init__(self, bridge):
        self.bridge=bridge
        self.lib=bridge.lib
        p=C.c_void_p;i=C.c_int
        self.lib.lsa_ward_scratch_bytes.argtypes=[]
        self.lib.lsa_ward_scratch_bytes.restype=C.c_size_t
        for name,args in {
            'lsa_ward_close_step':[p,p,i,p,p,p,p],
            'lsa_ward_close_range':[p,i,i,i,p,p,p,p],
            'lsa_ward_close_range_diagnostic':[p,i,i,i,p,p,p,p,p,p],
        }.items():
            getattr(self.lib,name).argtypes=args
            getattr(self.lib,name).restype=i
        assert self.lib.lsa_ward_scratch_bytes()==16
    def buffers(self,tokens):
        import torch
        blocks=(tokens+511)//512
        return dict(centroids=torch.full((blocks*64,64),-7.,device='cuda'),
                    members=torch.full((blocks*64,16),-7,dtype=torch.int32,device='cuda'),
                    scratch=torch.zeros(4,dtype=torch.int32,device='cuda'),
                    labels=torch.full((blocks,128),-7,dtype=torch.int32,device='cuda'),
                    objective=torch.full((blocks,),float('nan'),device='cuda'))
    def close(self,keys,b,first=0,count=None,diagnostic=True):
        count=len(keys)//512-first if count is None else count
        p=self.bridge.p
        arguments=[p(keys),len(keys),first,count,p(b['centroids']),p(b['members']),p(b['scratch'])]
        if diagnostic:
            arguments.extend([p(b['labels']),p(b['objective'])])
        self.bridge.call('lsa_ward_close_range_diagnostic' if diagnostic else 'lsa_ward_close_range',
                         *arguments,self.bridge.stream)
    def step(self,keys,position,b):
        p=self.bridge.p
        self.bridge.call('lsa_ward_close_step',p(keys),p(position),len(keys),p(b['centroids']),
                         p(b['members']),p(b['scratch']),self.bridge.stream)


def cpu_algorithm(keys):
    """Independent FP64 spherical Ward diagnostic, for small CPU-only cases."""
    import numpy as np
    pools=np.asarray(keys,dtype=np.float64).reshape(128,4,64).sum(1)
    alive=np.ones(128,dtype=bool);sizes=np.full(128,4,dtype=np.int32);parent=np.arange(128)
    rounds=0
    while int(alive.sum())>64:
        gram=pools@pools.T
        square=np.einsum('ij,ij->i',pools,pools)
        cost=np.sqrt(np.maximum(square,0))[:,None]+np.sqrt(np.maximum(square,0))[None,:]-np.sqrt(np.maximum(square[:,None]+square[None,:]+2*gram,0))
        allowed=alive[:,None]&alive[None,:]&(sizes[:,None]+sizes[None,:]<=16)
        np.fill_diagonal(allowed,False)
        cost=np.where(allowed,cost,np.inf)
        best=cost.argmin(1)
        candidates=[(float(cost[r,best[r]]),int(best[r]),r) for r in range(128)
                    if np.isfinite(cost[r,best[r]]) and r<int(best[r]) and int(best[best[r]])==r]
        candidates.sort()
        keep=candidates[:int(alive.sum())-64]
        if not keep:raise AssertionError('FP64 diagnostic Ward stalled')
        for _,partner,row in keep:
            pools[row]+=pools[partner];sizes[row]+=sizes[partner];alive[partner]=False;parent[partner]=row
        rounds+=1
        assert rounds<=128
    numbering={int(root):number for number,root in enumerate(np.flatnonzero(alive))}
    labels=[]
    for row in range(128):
        while parent[row]!=row:row=int(parent[row])
        labels.append(numbering[row])
    return dict(labels=labels,rounds=rounds,objective=float(np.linalg.norm(pools[alive],axis=1).sum()))


def cpu_prepare():
    import numpy as np
    # Independent algorithm/partition checks. No torch.cuda entry point is called.
    rng=np.random.default_rng(67411)
    normal=rng.normal(size=(512,64))
    normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    identical=np.zeros((512,64));identical[:,0]=1
    cases=[]
    for name,keys in [('normal',normal),('identical',identical)]:
        result=cpu_algorithm(keys)
        sizes=np.bincount(np.repeat(result['labels'],4),minlength=64)
        assert len(sizes)==64 and sizes.min()>=4 and sizes.max()<=16 and sizes.sum()==512
        cases.append(dict(case=name,rounds=result['rounds'],objective=result['objective']))
    return dict(status='cpu_prepared',gpu_executed=False,cases=cases,script_sha256=base.sha(__file__),
                source_sha256=base.sha(ROOT/'strata-src/src/lsa/ward.cu'),
                pinned_upstream='35b1bf9a10703b57da3eb073debfad6a66507269',
                api='ward_close_step/range;16-byte sticky diagnostics;no external algorithm scratch')


def objective64(keys,members):
    import torch
    keys=keys.cpu().double();members=members.cpu().long()
    alignment=[]
    for block in range(len(keys)//512):
        value=0.
        rows=members[block*64:(block+1)*64]
        for row in rows:
            valid=row[row>=0]
            value+=float(keys[valid].sum(0).norm())
        alignment.append(value)
    return torch.tensor(alignment,dtype=torch.float64)


def compare(ward,keys,name,allow_near_ties):
    import torch
    import torch.nn.functional as F
    from lsa.core.kernels.ward import cluster_ward
    from lsa.core.kernels.triton.ward import bucket_members_from_labels
    native=ward.buffers(len(keys))
    ward.close(keys,native)
    status=native['scratch'].cpu().tolist()
    if status[0]!=0:raise AssertionError(f'Native Ward failure{name}:diagnostics{status}')
    blocks=len(keys)//512
    assert status[1]==blocks and 1<=status[2]<=128
    pools=keys[:blocks*512].reshape(blocks,128,4,64).sum(2)
    labels,reached=cluster_ward(pools,torch.full((blocks,128),4.,device='cuda'),64,16,precision='ieee',kernels='triton')
    if not bool(reached.all()):raise AssertionError(f'Pinned upstream Ward stopped short in{name}')
    repeated=labels.repeat_interleave(4,1)
    expected_members=bucket_members_from_labels(repeated,64,16)
    expected_members=torch.where(expected_members>=0,expected_members+(torch.arange(blocks,device='cuda')*512)[:,None,None],-1).reshape(-1,16).int()
    actual_members=native['members'].cpu()
    actual_labels=native['labels'].cpu().long()
    expected_labels=labels.cpu()
    actual_centroids=native['centroids'].cpu()
    assert torch.isfinite(actual_centroids).all()
    assert torch.equal(actual_centroids,actual_centroids.bfloat16().float()),'Centroids are not BF16-rounded'
    for block in range(blocks):
        members=actual_members[block*64:(block+1)*64]
        valid=members[members>=0]
        assert torch.equal(valid.sort().values,torch.arange(block*512,(block+1)*512)),'Native partition lost/repeated keys'
        counts=(members>=0).sum(1)
        assert bool(((counts>=4)&(counts<=16)&(counts%4==0)).all())
        for row in members:
            live=row[row>=0]
            assert bool((live[1:]>live[:-1]).all()),'Members are not in original key order'
        reconstructed=[]
        for pool in range(128):
            matches=(members==block*512+pool*4).any(1).nonzero().flatten()
            assert len(matches)==1
            reconstructed.append(int(matches[0]))
        assert reconstructed==actual_labels[block].tolist()
    # Reconstruct both partitions' alignment from original key vectors in FP64.
    # Never relabel a different partition as exact parity.
    actual_objective=objective64(keys,actual_members)
    expected_objective=objective64(keys,expected_members)
    differences=(actual_objective-expected_objective).abs()
    relative=differences/expected_objective.abs().clamp_min(1e-12)
    changed=(actual_labels!=expected_labels).any(1)
    changed_blocks=changed.nonzero().flatten().tolist()
    if changed_blocks:
        if not allow_near_ties:
            raise AssertionError(f'Partition differs in{name} blocks{changed_blocks};FP64 alignment absdiff{differences.tolist()};explicit tie review required')
        assert float(differences.max())<=2e-4 and float(relative.max())<=5e-7, 'Different partition exceeds near-tie objective gate'
    # Every native centroid must equal its own partition's normalized key sum
    # within one BF16 rounding interval (not necessarily upstream's partition).
    expected_native=[]
    keys_cpu=keys.cpu()
    for row in actual_members:
        valid=row[row>=0].long()
        expected_native.append(F.normalize(keys_cpu[valid].sum(0),dim=0).bfloat16().float())
    expected_native=torch.stack(expected_native)
    centroid_delta=(actual_centroids-expected_native).abs()
    assert float(centroid_delta.max())<=0.004,'Centroid sums/normalization differ materially'
    if not changed_blocks:assert torch.equal(actual_members,expected_members.cpu()),'Labels agree but members differ'
    norms=actual_centroids.norm(dim=-1)
    assert bool((((norms>0.99)&(norms<1.01))|(norms==0)).all()), 'Centroid normalization invalid'
    result=dict(case=name,blocks=blocks,diagnostics=status,partition_exact=not changed_blocks,
                changed_blocks=changed_blocks,partition_complete_unique_and_capped=True,
                fp64_objective_native=actual_objective.tolist(),fp64_objective_upstream=expected_objective.tolist(),
                max_objective_abs_difference=float(differences.max()),max_objective_relative_difference=float(relative.max()),
                centroid_max_abs_difference=float(centroid_delta.max()),centroid_bf16_different_coordinates=int((actual_centroids!=expected_native).sum()),
                near_tie_policy='Only explicit --allow-near-ties: complete partition+FP64alignment abs<=2e-4 andrel<=5e-7; reported as alternative, never exact')
    return result,native


def graph_check(ward,stream,keys):
    import torch
    b=ward.buffers(len(keys));direct=ward.buffers(len(keys))
    pos=torch.tensor([510],dtype=torch.int32,device='cuda')
    ward.step(keys,pos,b);stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):ward.step(keys,pos,b)
    cases=[]
    for value in (510,511,512,1023,1024,1535,2047):
        before_centroids=b['centroids'].clone();before_members=b['members'].clone()
        pos.fill_(value);graph.replay()
        if value%512==511:
            block=value//512
            ward.close(keys,direct,block,1,False)
            sl=slice(block*64,(block+1)*64)
            assert torch.equal(b['members'][sl],direct['members'][sl])
            assert torch.equal(b['centroids'][sl],direct['centroids'][sl])
        else:
            assert torch.equal(b['members'],before_members) and torch.equal(b['centroids'],before_centroids)
        cases.append(dict(position=value,closed=value%512==511,graph_direct_exact=True))
    assert b['scratch'].cpu().tolist()[0]==0
    # Sticky error survives a later valid/nonboundary step and never masquerades
    # as a successful closure. Only this private test state is poisoned.
    pos.fill_(-1);ward.step(keys,pos,b)
    pos.fill_(512);ward.step(keys,pos,b)
    assert b['scratch'].cpu().tolist()[0]==-1
    short_cases=[]
    for capacity in (1,31,511):
        short=keys[:capacity].contiguous();small=ward.buffers(capacity)
        short_pos=torch.tensor([capacity-1],dtype=torch.int32,device='cuda')
        old_c=small['centroids'].clone();old_m=small['members'].clone()
        ward.step(short,short_pos,small)
        assert torch.equal(small['centroids'],old_c) and torch.equal(small['members'],old_m)
        assert small['scratch'].cpu().tolist()==[0,0,0,0]
        short_graph=torch.cuda.CUDAGraph();stream.synchronize()
        with torch.cuda.graph(short_graph,stream=stream):ward.step(short,short_pos,small)
        short_pos.fill_(0);short_graph.replay()
        assert torch.equal(small['centroids'],old_c) and torch.equal(small['members'],old_m)
        short_pos.fill_(511);ward.step(short,short_pos,small)
        assert small['scratch'].cpu().tolist()[0]==-1
        p=ward.bridge.p
        rc=ward.lib.lsa_ward_close_range(p(short),capacity,0,1,p(small['centroids']),p(small['members']),p(small['scratch']),ward.bridge.stream)
        assert rc==1,'Range admitted an incomplete512-key block'
        short_cases.append(dict(capacity=capacity,valid_step_and_graph_noop=True,invalid_boundary_rejected=True,incomplete_range_refused=True))
    return dict(capture_passed=True,cases=cases,sticky_invalid_position_error=True,small_context_checks=short_cases)


def worker(args):
    if os.environ.get('LSA_SUPERVISED_OWNER')!=args.owner or args.owner!=base.EXPECTED_OWNER:
        raise RuntimeError('Worker requires current bounded maintenance supervisor')
    if base.service_state()!='inactive':raise RuntimeError('Production is not inactive')
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(4096*1048576/torch.cuda.get_device_properties(0).total_memory,0)
    sys.path.insert(0,str(base.RESEARCH/'upstream-research/src'))
    stream=torch.cuda.Stream();bridge=base.Bridge(stream,args.library);ward=Ward(bridge)
    error=C.create_string_buffer(1024)
    handle=bridge.lib.lsa_maps_open(os.fsencode(base.NATIVE_MAPS),error,len(error))
    if not handle:raise RuntimeError(error.value.decode())
    report=dict(status='running',started_at=base.utc(),gpu_executed=True,cases=[],
                scope='Native512-key spherical Ward block closure component qualification',
                source_sha256=base.sha(ROOT/'strata-src/src/lsa/ward.cu'),library_sha256=base.sha(args.library),
                map_sha256=base.sha(base.NATIVE_MAPS),script_sha256=base.sha(__file__),
                allow_near_ties=args.allow_near_ties,
                upstream_commit=base.command(['git','-C',str(base.RESEARCH/'upstream-research'),'rev-parse','HEAD']).strip(),
                limitations=['No whole-model quality or speed claim','FP32Gram rounding may differ from Triton IEEE dot; different partitions are explicit'])
    if report['upstream_commit']!='35b1bf9a10703b57da3eb073debfad6a66507269':raise RuntimeError('Pinned upstream source changed')
    try:
        with torch.cuda.stream(stream):
            for layer in base.LAYERS:
                if base.service_state()!='inactive':raise RuntimeError('Production state changed')
                data=base.load_capture(base.MAPROOT/'captures/test_code'/f'layer-{layer}-p0.lsacap')
                pools=base.paged_pools(data,seed=1877+layer)
                history=torch.empty((data['tokens'],64),device='cuda')
                weight=bridge.lib.lsa_maps_key(handle,layer)
                assert weight
                bridge.project_keys(pools,weight,history)
                result,_=compare(ward,history,f'real-layer-{layer}',args.allow_near_ties)
                result['capture_sha256']=data['sha256']
                report['cases'].append(result)
                print(json.dumps(dict(event='ward_case',**result)),flush=True)
                del data,pools,history
            gen=torch.Generator(device='cuda').manual_seed(75125)
            random=F.normalize(torch.randn((2048,64),generator=gen,device='cuda'),dim=-1)
            result,_=compare(ward,random,'synthetic-random',args.allow_near_ties);report['cases'].append(result)
            report['graphs']=graph_check(ward,stream,random)
            identical=torch.zeros((512,64),device='cuda');identical[:,0]=1
            result,_=compare(ward,identical,'synthetic-identical',args.allow_near_ties);report['cases'].append(result)
            near=identical+torch.randn((512,64),generator=gen,device='cuda')*1e-3
            result,_=compare(ward,F.normalize(near,dim=-1),'synthetic-near-copy',args.allow_near_ties);report['cases'].append(result)
            stream.synchronize()
        report.update(status='passed',peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,
                      exact_cases=sum(c['partition_exact'] for c in report['cases']),alternative_cases=sum(not c['partition_exact'] for c in report['cases']))
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        bridge.lib.lsa_maps_close(handle)
        report['finished_at']=base.utc();base.write_exclusive(args.output,report)


def execute(args):
    before=base.maintenance_state(args.owner);gpu=base.gpu_snapshot()
    utilization_limit=20 if args.allow_desktop_load else 5
    compute_processes=[]
    if args.allow_desktop_load:
        import csv
        compute_processes=list(csv.DictReader(base.command(['nvidia-smi','--query-compute-apps=process_name','--format=csv']).splitlines()))
        if any(row.get('process_name','').strip()!='/usr/bin/sunshine' for row in compute_processes):raise RuntimeError('Desktop allowance requires empty computeapps or Sunshine only')
    if gpu['capability']!='12.0' or gpu['free_mib']<8192 or gpu['utilization']>utilization_limit:raise RuntimeError('GPU window exceeds correctness preflight limits')
    run=ROOT/'receipts'/('ward-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-')+uuid.uuid4().hex[:8]);run.mkdir(exist_ok=False)
    env=dict(os.environ);env.pop('NVIDIA_VISIBLE_DEVICES',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',LSA_SUPERVISED_OWNER=args.owner,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for key,name in [('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TORCH_EXTENSIONS_DIR','extensions'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('XDG_CACHE_HOME','xdg')]:
        p=ROOT/'cache'/name;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
    output=run/'result.json'
    argv=[str(base.PYTHON),str(Path(__file__).resolve()),'--worker','--owner',args.owner,'--output',str(output),'--library',str(args.library)]
    if args.allow_near_ties:argv.append('--allow-near-ties')
    receipt=dict(started_at=base.utc(),status='launching',before=before,gpu=gpu,timeout_seconds=args.max_seconds,argv=argv,
                 desktop_load_allowed=args.allow_desktop_load,preflight_utilization_limit=utilization_limit,
                 compute_processes=compute_processes,performance_claim=False)
    proc=None
    try:
        with (run/'worker.log').open('x') as log:
            base.maintenance_state(args.owner)
            proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+args.max_seconds;receipt['pid']=proc.pid
            while True:
                remain=deadline-time.monotonic()
                if remain<=0:raise TimeoutError('Ward worker external deadline')
                try:rc=proc.wait(timeout=min(5,remain));break
                except subprocess.TimeoutExpired:base.maintenance_state(args.owner,timeout=min(15,max(.1,deadline-time.monotonic())))
        receipt['after']=base.maintenance_state(args.owner);receipt['returncode']=rc
        if rc!=0 or not output.is_file():raise RuntimeError(f'Ward worker exited{rc}; retained worker.log')
        result=json.loads(output.read_text())
        if result.get('status')!='passed':raise RuntimeError('Ward result failed')
        receipt.update(status='ward_component_passed',result_sha256=base.sha(output))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:base.kill_owned(proc)
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        receipt['finished_at']=base.utc();base.write_exclusive(run/'receipt.json',receipt)
        print(json.dumps(dict(status=receipt['status'],receipt=str(run/'receipt.json'))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group()
    m.add_argument('--prepare',action='store_true');m.add_argument('--execute',action='store_true');m.add_argument('--worker',action='store_true')
    p.add_argument('--owner');p.add_argument('--library',type=Path,default=base.LIBRARY);p.add_argument('--output',type=Path)
    p.add_argument('--max-seconds',type=int,default=1200)
    p.add_argument('--allow-desktop-load',action='store_true',help='Correctness only: allow<=20percent if computeapps empty or Sunshine only')
    p.add_argument('--allow-near-ties',action='store_true',help='Explicit objective/partition gate for reported numerically different alternatives')
    a=p.parse_args()
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal{signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
    if not 60<=a.max_seconds<=1800:p.error('max-seconds60..1800')
    if a.worker and not a.output:p.error('worker requires output')
    if a.prepare:
        value=cpu_prepare()
        if a.output:base.write_exclusive(a.output,value)
        print(json.dumps(value,indent=2))
    elif a.worker:worker(a)
    elif a.execute:execute(a)
    else:print(json.dumps(dict(mode='plan',gpu_executed=False,scope='Native Ward component; GPU execution must be coordinated with root')))

if __name__=='__main__':main()
