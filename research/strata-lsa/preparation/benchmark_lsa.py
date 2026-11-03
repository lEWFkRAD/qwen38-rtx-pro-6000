"""Bounded synthetic LSA indexer diagnostic; GPU execution requires explicit opt-in.
No model accuracy, full-context prefill, or end-to-end serving speed claim.
Use run_window.py for source/resource guards and an external hard timeout.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback
SOURCE_SHA='906be195c1052aaa74f95852ad1917ce1926ba64'
MAX_CONTEXT=1048576
DIM,BLOCK,POOL,BUCKET,CAP=64,512,4,8,16
TOP_P,TOP_K,SINK,WINDOW=512,2048,4,512

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','cpu','gpu'),default='plan')
    p.add_argument('--allow-gpu',action='store_true')
    p.add_argument('--upstream',type=Path,default=Path(__file__).parent/'upstream')
    p.add_argument('--contexts',nargs='+',type=int,default=[131072,262144,1048576])
    p.add_argument('--query-batch',type=int,default=64)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmups',type=int,default=1)
    p.add_argument('--max-gpu-mib',type=int,default=2048)
    p.add_argument('--max-seconds',type=int,default=900)
    p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--output',type=Path)
    return p

def validate(a):
    if a.mode=='gpu' and not a.allow_gpu:
        raise ValueError('GPU execution requires --mode gpu and --allow-gpu')
    if a.allow_gpu and a.mode!='gpu':
        raise ValueError('--allow-gpu requires --mode gpu')
    if not 1<=len(a.contexts)<=4 or len(set(a.contexts))!=len(a.contexts):
        raise ValueError('One to four distinct contexts required')
    if any(n<1024 or n>MAX_CONTEXT or n%BLOCK for n in a.contexts):
        raise ValueError('Contexts must be multiples of512 in [1024,1048576]')
    if a.mode=='gpu' and min(a.contexts)<8192:
        raise ValueError('GPU timing contexts must be at least8192 for fixed512-bucket selection')
    for name,lo,hi in [('query_batch',1,128),('repeats',1,10),('warmups',1,3),('max_gpu_mib',512,8192),('max_seconds',1,1800)]:
        if not lo<=getattr(a,name)<=hi:
            raise ValueError(f'{name} must be in [{lo},{hi}]')

def plan(a):
    return dict(source_sha_expected=SOURCE_SHA,backend='triton',precision='ieee',dtype='fp32 keys / bf16 scoring',contexts=a.contexts,index_dim=DIM,cluster_block=BLOCK,pool_size=POOL,bucket_size=BUCKET,bucket_cap=CAP,top_p=TOP_P,top_k=TOP_K,sink=SINK,window=WINDOW,query_batches=sorted(set((1,a.query_batch))),repeats=a.repeats,warmups=a.warmups,max_gpu_mib=a.max_gpu_mib,max_seconds=a.max_seconds,largest_score_matrix_elements=max(a.contexts)//BUCKET*a.query_batch,scope='Synthetic flat index construction, selection and expansion',limitations=['No model accuracy, memory fit or serving compatibility claim','No full-context prefill or model speedup measured','External hard timeout and source verification required','Small CPU smoke is not a passing upstream baseline','Known upstream near-copy failures are retained separately'])

def summary(samples):
    if not samples or not all(math.isfinite(x) and x>=0 for x in samples):
        raise ValueError('Empty, negative or nonfinite timing samples')
    return dict(samples_ms=samples,median_ms=statistics.median(samples),min_ms=min(samples),max_ms=max(samples))

def check_rows(rows,lengths,positions):
    if len(rows)!=len(lengths) or len(rows)!=len(positions):
        raise AssertionError('Row, length and position counts differ')
    for row,length,position in zip(rows,lengths,positions):
        if not 0<=length<=len(row):
            raise AssertionError('Invalid selected length')
        active=row[:length]
        if any(key<0 or key>position for key in active):
            raise AssertionError('Noncausal or negative selected key')
        if any(key!=-1 for key in row[length:]):
            raise AssertionError('Output padding is not -1')
        if len(set(active))!=len(active):
            raise AssertionError('Duplicate selected key')
        window=set(range(max(0,position+1-WINDOW),position+1))
        sink=set(range(min(SINK,position+1)))
        if not window.union(sink).issubset(set(active)):
            raise AssertionError('Sink or recent-window key missing')
    return dict(rows=len(rows),causal_unique_window_sink_padding=True)

def load_reference(a):
    path=a.upstream.resolve()/'src'
    if not (path/'lsa/core/reference.py').is_file():
        raise FileNotFoundError(f'Upstream missing: {path}')
    sys.path.insert(0,str(path))
    import torch
    from lsa.core.reference import Buckets,build_buckets,read,expand_buckets
    if torch.__version__.split('+')[0]!='2.13.0':
        raise RuntimeError(f'Pinned torch2.13.0 required, found {torch.__version__}')
    torch.set_num_threads(2)
    return torch,Buckets,build_buckets,read,expand_buckets

def cpu_smoke(a,report):
    torch,_,build_buckets,read,_=load_reference(a)
    gen=torch.Generator(device='cpu').manual_seed(a.seed)
    keys=torch.nn.functional.normalize(torch.randn(2048,DIM,generator=gen),dim=-1)
    queries=torch.nn.functional.normalize(torch.randn(6,DIM,generator=gen),dim=-1)
    positions=torch.tensor([0,3,511,512,1023,2047],dtype=torch.int32)
    start=time.perf_counter()
    buckets=build_buckets(keys,POOL,BUCKET,CAP,BLOCK)
    build_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    selected,lengths=read(queries,positions,buckets,TOP_P,TOP_K,SINK,WINDOW)
    read_ms=(time.perf_counter()-start)*1000
    checks=check_rows(selected.tolist(),lengths.tolist(),positions.tolist())
    members=buckets.bucket_members.flatten()
    if sorted(members[members>=0].tolist())!=list(range(2048)):
        raise AssertionError('Cluster partition invalid')
    again=read(queries,positions,buckets,TOP_P,TOP_K,SINK,WINDOW)
    if not torch.equal(again[0],selected) or not torch.equal(again[1],lengths):
        raise AssertionError('Reference selection nondeterministic')
    report.update(torch_version=torch.__version__,status='passed',gpu_executed=False,cpu_smoke=dict(keys_shape=[2048,DIM],query_count=6,build_ms=build_ms,read_ms=read_ms,checks={**checks,'partition':True,'repeat':True}),qualification_claim=False)

class Deadline:
    def __init__(self,seconds):
        self.end=time.monotonic()+seconds
    def check(self):
        if time.monotonic()>=self.end:
            raise TimeoutError('Cooperative benchmark budget exhausted')

def nearcopy_qualification(torch,cluster_ward,ward_rounds,seed):
    # Exact near-copy seed1 fixtures of the failed upstream CPU tests.
    # An incomplete CPU reference is reported, never silently replaced by FP64.
    results=[]
    for pool in (1,4):
        gen=torch.Generator(device='cpu').manual_seed(1)
        keys=torch.nn.functional.normalize(torch.randn(8,128*pool,16,generator=gen),dim=-1)
        pools=keys.view(8,128,pool,16).sum(2)
        pools=pools[:,:1].expand_as(pools)+1e-3*pools
        sizes=torch.full((8,128),float(pool))
        target=128*pool//8
        row=dict(pool_size=pool,seed=1,index_dim=16,target_buckets=target,cap=16)
        try:
            want,reference_reached=ward_rounds(pools,sizes,target,16)
            got,reached=cluster_ward(pools.cuda(),sizes.cuda(),target,16,precision='ieee',kernels='triton')
            got,reached=got.cpu(),reached.cpu()
            row.update(reference_reached=reference_reached.tolist(),kernel_reached=reached.tolist())
            if not bool(reference_reached.all()):
                row['status']='reference_failed'
            elif not bool(reached.all()):
                row['status']='failed'
            else:
                row['exact_labels']=torch.equal(got,want)
                row['status']='passed' if row['exact_labels'] else 'failed'
        except Exception as exc:
            row.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)))
        results.append(row)
    return results

def gpu_run(a,report):
    torch,Buckets,_,_,expand_buckets=load_reference(a)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    capability=torch.cuda.get_device_capability(0)
    if capability!=(12,0):
        raise RuntimeError(f'Kit targets sm120 only, found {capability}')
    prop=torch.cuda.get_device_properties(0)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(a.max_gpu_mib*1024**2/prop.total_memory,0)
    torch.backends.cuda.matmul.allow_tf32=False
    report.update(gpu_executed=True,torch_version=torch.__version__,gpu=dict(name=prop.name,capability=list(capability),total_memory_bytes=prop.total_memory),dependencies={name:importlib.metadata.version(name) for name in ('triton','nvidia-cutlass-dsl','apache-tvm-ffi')},qualification_claim=False)
    from lsa.core.kernels.ward import cluster_ward
    from lsa.core.kernels.triton.ward import bucket_members_from_labels
    from lsa.core.kernels.scan_select import ScanSelect,dense_keys
    from lsa.core.kernels.triton.keys import unpack_ids
    from lsa.core.kernels.expansion import expand_in_order
    from lsa.core.ward import ward_rounds
    deadline=Deadline(a.max_seconds)
    report['cases']=[]
    def timed(fn):
        deadline.check()
        torch.cuda.synchronize()
        start=time.perf_counter()
        result=fn()
        torch.cuda.synchronize()
        return result,(time.perf_counter()-start)*1000
    def measure(fn):
        out,first=timed(fn)
        warm=[]
        for _ in range(a.warmups):
            out,ms=timed(fn)
            warm.append(ms)
        samples=[]
        for _ in range(a.repeats):
            out,ms=timed(fn)
            samples.append(ms)
        return out,dict(first_call_ms_including_possible_jit=first,warmup_ms=warm,**summary(samples))
    gen=torch.Generator(device='cpu').manual_seed(a.seed)
    small=torch.nn.functional.normalize(torch.randn(4,BLOCK,DIM,generator=gen),dim=-1).cuda()
    pools=small.view(4,BLOCK//POOL,POOL,DIM).sum(2)
    sizes=torch.full((4,BLOCK//POOL),float(POOL),device='cuda')
    (got,reached),timing=timed(lambda:cluster_ward(pools,sizes,BLOCK//BUCKET,CAP,precision='ieee',kernels='triton'))
    want,want_reached=ward_rounds(pools.cpu(),sizes.cpu(),BLOCK//BUCKET,CAP)
    if not bool(reached.all()) or not bool(want_reached.all()):
        raise AssertionError('Small IID Ward did not complete')
    if not torch.equal(got.cpu(),want):
        raise AssertionError('Small IID Ward CPU exact-label oracle failed')
    report['ward_smoke']=dict(exact_labels=True,reached=True,first_call_ms_including_possible_jit=timing)
    del small,pools,sizes,got,reached,want,want_reached
    report['nearcopy_qualification']=nearcopy_qualification(torch,cluster_ward,ward_rounds,a.seed)
    if any(row['status']!='passed' for row in report['nearcopy_qualification']):
        report.update(status='qualification_failed',qualification_note='Known near-copy reference/parity failure retained. Performance phase not started; review or repair reference in a separately versioned change.')
        return
    for context in a.contexts:
        deadline.check()
        torch.cuda.reset_peak_memory_stats()
        gen=torch.Generator(device='cpu').manual_seed(a.seed+context)
        keys_cpu=torch.nn.functional.normalize(torch.randn(context,DIM,generator=gen),dim=-1)
        keys=keys_cpu.cuda()
        del keys_cpu
        case=dict(context=context,keys_shape=[context,DIM],reads=[])
        report['cases'].append(case)
        block_count,per_block=context//BLOCK,BLOCK//BUCKET
        def build():
            centroids=torch.empty(context//BUCKET,DIM,device='cuda',dtype=torch.bfloat16)
            members=torch.empty(context//BUCKET,CAP,device='cuda',dtype=torch.int32)
            for first in range(0,block_count,32):
                deadline.check()
                count=min(32,block_count-first)
                block=keys[first*BLOCK:(first+count)*BLOCK].view(count,BLOCK,DIM)
                pooled=block.view(count,BLOCK//POOL,POOL,DIM).sum(2)
                sizes=torch.full((count,BLOCK//POOL),float(POOL),device='cuda')
                labels,reached=cluster_ward(pooled,sizes,per_block,CAP,precision='ieee',kernels='triton')
                if not bool(reached.all()):
                    raise AssertionError('Ward did not reach bucket count')
                labels=labels.repeat_interleave(POOL,dim=1)
                sums=torch.zeros(count,per_block,DIM,device='cuda')
                sums.scatter_add_(1,labels[...,None].expand_as(block),block)
                batch_members=bucket_members_from_labels(labels,per_block,CAP)
                offsets=(torch.arange(first,first+count,device='cuda')*BLOCK)[:,None,None]
                batch_members=torch.where(batch_members>=0,batch_members+offsets,-1)
                sl=slice(first*per_block,(first+count)*per_block)
                centroids[sl]=torch.nn.functional.normalize(sums,dim=-1).reshape(-1,DIM).bfloat16()
                members[sl]=batch_members.reshape(-1,CAP)
            if not bool(torch.isfinite(centroids).all()):
                raise AssertionError('Nonfinite centroid')
            return centroids,members
        (centroids,members),case['build']=measure(build)
        del keys
        cpu_members=members.cpu()
        cpu_centroids=centroids.float().cpu()
        buckets_cpu=Buckets(cpu_centroids,cpu_members,(cpu_members>=0).sum(1).int(),per_block,BLOCK)
        for rows in sorted(set((1,a.query_batch))):
            deadline.check()
            queries_cpu=torch.nn.functional.normalize(torch.randn(rows,DIM,generator=gen),dim=-1).bfloat16()
            positions=torch.arange(context-rows,context,dtype=torch.int32,device='cuda')
            visible=(positions//BLOCK*per_block).contiguous()
            queries=queries_cpu.cuda()
            select=ScanSelect(rows,TOP_P,device='cuda',kernels='triton')
            picked,selection_timing=measure(lambda:select(queries,centroids,visible))
            exact=unpack_ids(dense_keys(queries,centroids,visible,kernels='triton').topk(TOP_P,dim=1).values)
            if not torch.equal(picked,exact):
                raise AssertionError('Packed-score selector oracle failed')
            picked_cpu=picked.cpu().long()
            visible_cpu=visible.cpu().tolist()
            score_rows=queries_cpu.float()@cpu_centroids.T
            if not bool(torch.isfinite(score_rows).all()):
                raise AssertionError('Nonfinite selection scores')
            minimum_margin=1.0
            for row,visible_n in enumerate(visible_cpu):
                valid=picked_cpu[row][picked_cpu[row]>=0]
                if valid.numel()!=min(visible_n,TOP_P) or valid.unique().numel()!=valid.numel():
                    raise AssertionError('Invalid/duplicate selector output')
                if bool((valid>=visible_n).any()):
                    raise AssertionError('Selector read future bucket')
                if valid.numel():
                    cutoff=score_rows[row,:visible_n].topk(min(TOP_P,visible_n)).values[-1]
                    margin=float(score_rows[row,valid].min()-cutoff)
                    minimum_margin=min(minimum_margin,margin)
                    if not math.isfinite(margin) or margin < -1e-5:
                        raise AssertionError(f'CPU selection cutoff violated: {margin}')
            del score_rows,exact
            (selected,lengths),expansion_timing=measure(lambda:expand_in_order(picked,members,positions,WINDOW,SINK,TOP_K,kernels='triton'))
            selected_cpu,lengths_cpu,positions_cpu=selected.cpu(),lengths.cpu(),positions.cpu()
            for row,position in enumerate(positions_cpu.tolist()):
                chosen=picked_cpu[row][picked_cpu[row]>=0]
                expected=expand_buckets(chosen,position,buckets_cpu,TOP_K,SINK,WINDOW)
                n=int(lengths_cpu[row])
                if n!=expected.numel() or not torch.equal(selected_cpu[row,:n],expected):
                    raise AssertionError('Expansion differs from CPU reference for identical picks')
            checks=check_rows(selected_cpu.tolist(),lengths_cpu.tolist(),positions_cpu.tolist())
            case['reads'].append(dict(query_count=rows,selection=selection_timing,expansion=expansion_timing,checks={**checks,'packed_score_oracle_exact':True,'cpu_score_tolerance':1e-5,'cpu_score_min_margin':minimum_margin,'expansion_reference_exact':True}))
            del select,queries,positions,visible,picked,picked_cpu,selected,lengths
        case['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        case['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        del centroids,members,cpu_members,cpu_centroids,buckets_cpu
    report['status']='passed'

def main(argv=None):
    a=parser().parse_args(argv)
    report=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),mode=a.mode,status='not_run',gpu_executed=False,qualification_claim=False)
    code=0
    try:
        validate(a)
        report['plan']=plan(a)
        if a.mode=='plan':
            report['status']='planned_only'
        elif a.mode=='cpu':
            cpu_smoke(a,report)
        else:
            gpu_run(a,report)
        if report['status']=='qualification_failed':
            code=1
    except Exception as exc:
        report.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        code=1
    data=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x',encoding='utf-8') as handle:
            handle.write(data)
    print(data,end='')
    return code
if __name__=='__main__':
    raise SystemExit(main())
