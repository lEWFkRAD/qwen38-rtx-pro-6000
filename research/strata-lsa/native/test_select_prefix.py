#!/usr/bin/env python3
"""Exact single-row visible-prefix sort versus unchanged two-row capacity sort.
Default is a plan. GPU worker execution belongs to the coordinated supervisor.
"""
from __future__ import annotations
import argparse
import ctypes as C
import json
import statistics
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
import test_cuda_adapter as base


def parity(bridge, stream):
    import torch
    import torch.nn.functional as F
    buckets=16384
    generator=torch.Generator(device='cuda').manual_seed(34919)
    centroids=F.normalize(torch.randn((buckets,64),generator=generator,device='cuda'),dim=-1).bfloat16().float().contiguous()
    query=F.normalize(torch.randn((1,64),generator=generator,device='cuda'),dim=-1).bfloat16().float().contiguous()
    positions=torch.zeros(1,dtype=torch.int32,device='cuda')
    members=torch.full((buckets,16),-1,dtype=torch.int32,device='cuda')
    members[:,:8]=torch.arange(buckets*8,dtype=torch.int32,device='cuda').reshape(-1,8)
    single=bridge.buffers(1,buckets)
    control=bridge.buffers(2,buckets)
    query2=query.repeat(2,1).contiguous()
    positions2=positions.repeat(2).contiguous()
    def one():bridge.select(query,centroids,members,positions,single)
    def two():bridge.select(query2,centroids,members,positions2,control)
    one();two();stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):one()
    checked=[]
    # Descending then ascending contexts also catch stale output after an empty
    # segment follows a previous full-capacity sort in the same graph buffers.
    values=[131071,0,-1,511,512,513,1023,1024,1535,1536,4096,65535,131071,131072,0]
    for value in values:
        positions.fill_(value);positions2.fill_(value)
        graph.replay();two()
        assert torch.equal(single['selected'][0],control['selected'][0]),('bucket order',value)
        assert torch.equal(single['cells'][0],control['cells'][0]),('cell order',value)
        assert torch.equal(single['steps'][0],control['steps'][0]),('step values',value)
        visible=min(value//512*64,buckets) if 0<=value<131072 else 0
        assert single['offsets'].cpu().tolist()==[0,visible],('single offsets',value)
        assert control['offsets'].cpu().tolist()==[0,buckets,2*buckets]
        if visible:assert torch.equal(single['keys_out'][0,:visible],control['keys_out'][0,:visible])
        graph_outputs={key:single[key].clone() for key in ('selected','cells','steps')}
        one()
        for key,value_tensor in graph_outputs.items():assert torch.equal(value_tensor,single[key]),('graph/direct',key,value)
        checked.append(dict(position=value,visible_buckets=visible,exact_capacity_control=True,graph_direct_exact=True))
    query.zero_();query2.zero_()
    for value in (512,131071,0):
        positions.fill_(value);positions2.fill_(value);graph.replay();two()
        for key in ('selected','cells','steps'):assert torch.equal(single[key][0],control[key][0]),('equal-score control',key,value)
    stream.synchronize()
    return dict(status='passed',gpu_executed=True,capacity_buckets=buckets,capacity_tokens=131072,
                cases=checked,zero_query_ties_exact=True,performance_claim=False,
                reference='Unchanged multirow capacity-strided segmented sort with identical per-row inputs')



def graph_launch_counts(bridge,graph):
    # CUDA runtime introspection counts captured kernel nodes rather than
    # guessing how many launches the current CUB implementation expands into.
    get_nodes=bridge.lib.cudaGraphGetNodes
    get_nodes.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)]
    get_nodes.restype=C.c_int
    get_type=bridge.lib.cudaGraphNodeGetType
    get_type.argtypes=[C.c_void_p,C.POINTER(C.c_int)]
    get_type.restype=C.c_int
    raw=C.c_void_p(graph.raw_cuda_graph());count=C.c_size_t()
    rc=get_nodes(raw,None,C.byref(count))
    if rc:raise RuntimeError(f'cudaGraphGetNodes returned{rc}')
    nodes=(C.c_void_p*count.value)()
    rc=get_nodes(raw,nodes,C.byref(count))
    if rc:raise RuntimeError(f'cudaGraphGetNodes returned{rc}')
    kinds={}
    for node in nodes:
        kind=C.c_int();rc=get_type(node,C.byref(kind))
        if rc:raise RuntimeError(f'cudaGraphNodeGetType returned{rc}')
        kinds[str(kind.value)]=kinds.get(str(kind.value),0)+1
    return dict(kernel_launches=kinds.get('0',0),total_graph_nodes=count.value,node_type_counts=kinds,
                kernel_node_count_source='cudaGraphGetNodes/cudaGraphNodeGetType')


def measure(bridge,stream,repeats):
    import torch
    import torch.nn.functional as F
    started=time.monotonic();deadline=started+25
    generator=torch.Generator(device='cuda').manual_seed(36191)
    all_centroids=F.normalize(torch.randn((16384,64),generator=generator,device='cuda'),dim=-1).bfloat16().float().contiguous()
    all_members=torch.full((16384,16),-1,dtype=torch.int32,device='cuda')
    all_members[:,:8]=torch.arange(131072,dtype=torch.int32,device='cuda').reshape(-1,8)
    results=[]
    for queries in (1,8):
        for requested in (0,192,1024,4096,16384):
            capacity=16384 if queries==1 else max(64,requested)
            # No valid131072-token request sees the last closed512-token block:
            # it remains in the recent window. Do not time invalidposition131072
            # and accidentally report its empty segment as full-context timing.
            position=min(requested//64*512,131071) if requested else 0
            visible=min(position//512*64,capacity)
            query=F.normalize(torch.randn((queries,64),generator=generator,device='cuda'),dim=-1).bfloat16().float().contiguous()
            positions=torch.full((queries,),position,dtype=torch.int32,device='cuda')
            centroids=all_centroids[:capacity];members=all_members[:capacity]
            buffers=bridge.buffers(queries,capacity)
            def operation():bridge.select(query,centroids,members,positions,buffers)
            for _ in range(3):operation()
            # External event nodes time inside the graph and avoid measuring
            # Python gaps between separately submitted start/select/end calls.
            begin=torch.cuda.Event(enable_timing=True,external=True)
            end=torch.cuda.Event(enable_timing=True,external=True)
            begin.record(stream);end.record(stream);stream.synchronize()
            graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph,stream=stream):
                begin.record(stream);operation();end.record(stream)
            graph.instantiate()
            launches=graph_launch_counts(bridge,graph)
            for _ in range(3):graph.replay()
            end.synchronize()
            samples=[]
            for _ in range(repeats):
                if time.monotonic()>deadline:raise TimeoutError('Optional selector timing exceeded25seconds')
                graph.replay();end.synchronize();samples.append(float(begin.elapsed_time(end)))
            results.append(dict(queries=queries,requested_bucket_count=requested,capacity_buckets=capacity,
                                actual_visible_buckets_per_row=visible,position=position,
                                warm_repeats=repeats,median_ms=statistics.median(samples),min_ms=min(samples),max_ms=max(samples),
                                samples_ms=samples,graph_replays_per_sample=1,**launches))
    return dict(scope='Captured native lsa_select only',cuda_event_method='External timing events inside captured graph',
                whole_model_speed_claim=False,elapsed_seconds=time.monotonic()-started,cases=results,
                maximum_context_note='Requested16384 capacity corresponds to16320 visible buckets at final validposition131071')


def worker(args):
    if os.environ.get('LSA_SUPERVISED_OWNER')!=args.owner or args.owner!=base.EXPECTED_OWNER:raise RuntimeError('Owned supervisor required')
    if base.service_state()!='inactive':raise RuntimeError('Production is not inactive')
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(4096*1048576/torch.cuda.get_device_properties(0).total_memory,0)
    stream=torch.cuda.Stream();bridge=base.Bridge(stream,args.library)
    with torch.cuda.stream(stream):
        report=parity(bridge,stream)
        if args.timings:report['timings']=measure(bridge,stream,args.timing_repeats)
    report.update(started_at=args.started_at,finished_at=base.utc(),script_sha256=base.sha(__file__),library_sha256=base.sha(args.library))
    base.write_exclusive(args.output,report)
    print(json.dumps(report),flush=True)


def execute(args):
    before=base.maintenance_state(args.owner);gpu=base.gpu_snapshot()
    limit=20 if args.allow_desktop_load else 5
    if args.allow_desktop_load:
        import csv
        processes=list(csv.DictReader(base.command(['nvidia-smi','--query-compute-apps=process_name','--format=csv']).splitlines()))
        if any(row.get('process_name','').strip()!='/usr/bin/sunshine' for row in processes):raise RuntimeError('Desktop allowance requires empty computeapps or Sunshine only')
    if gpu['capability']!='12.0' or gpu['free_mib']<8192 or gpu['utilization']>limit:raise RuntimeError('GPU preflight refused')
    run=base.ROOT/'receipts'/('select-prefix-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-')+uuid.uuid4().hex[:8]);run.mkdir(exist_ok=False)
    env=dict(os.environ);env.pop('NVIDIA_VISIBLE_DEVICES',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',LSA_SUPERVISED_OWNER=args.owner,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for key,name in [('CUDA_CACHE_PATH','cuda'),('TRITON_CACHE_DIR','triton'),('TORCH_EXTENSIONS_DIR','extensions'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('XDG_CACHE_HOME','xdg')]:
        cache=base.ROOT/'cache'/name;cache.mkdir(parents=True,exist_ok=True);env[key]=str(cache)
    argv=[str(base.PYTHON),str(base.ROOT/'test_select_prefix.py'),'--worker','--owner',args.owner,'--library',str(args.library),'--output',str(run/'result.json'),'--started-at',base.utc()]
    if args.timings:argv.extend(['--timings','--timing-repeats',str(args.timing_repeats)])
    receipt=dict(started_at=base.utc(),status='launching',before=before,gpu=gpu,argv=argv,performance_claim=False,desktop_load_allowed=args.allow_desktop_load)
    proc=None
    try:
        with (run/'worker.log').open('x') as log:
            base.maintenance_state(args.owner)
            proc=subprocess.Popen(argv,cwd=base.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+args.max_seconds
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Visible-prefix sort worker deadline')
                try:rc=proc.wait(timeout=min(5,remaining));break
                except subprocess.TimeoutExpired:base.maintenance_state(args.owner,timeout=min(15,max(.1,deadline-time.monotonic())))
        receipt['after']=base.maintenance_state(args.owner)
        if rc!=0:raise RuntimeError(f'Worker exit{rc}; retained worker.log')
        result=json.loads((run/'result.json').read_text())
        if result.get('status')!='passed':raise RuntimeError('Worker did not pass')
        receipt.update(status='passed',result_sha256=base.sha(run/'result.json'))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:base.kill_owned(proc)
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        receipt['finished_at']=base.utc();base.write_exclusive(run/'receipt.json',receipt)
        print(json.dumps(dict(status=receipt['status'],receipt=str(run/'receipt.json'))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group()
    m.add_argument('--worker',action='store_true');m.add_argument('--execute',action='store_true')
    p.add_argument('--owner');p.add_argument('--output',type=base.Path);p.add_argument('--library',type=base.Path,default=base.LIBRARY)
    p.add_argument('--timings',action='store_true',help='Bounded captured CUDAevent selector timings after exactparity passes')
    p.add_argument('--timing-repeats',type=int,default=12)
    p.add_argument('--started-at',default='');p.add_argument('--allow-desktop-load',action='store_true');p.add_argument('--max-seconds',type=int,default=180)
    a=p.parse_args()
    if a.worker and a.output is None:p.error('worker requires output')
    if not 60<=a.max_seconds<=600:p.error('max-seconds60..600')
    if not 1<=a.timing_repeats<=20:p.error('timing-repeats1..20')
    def interrupt(signum,frame):raise KeyboardInterrupt(f'Signal{signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGHUP,interrupt)
    if a.worker:worker(a)
    elif a.execute:execute(a)
    else:print(json.dumps(dict(mode='plan',gpu_executed=False,test='Single-row dynamic visible-prefix sort against unchanged two-row full-capacity reference')))

if __name__=='__main__':main()
