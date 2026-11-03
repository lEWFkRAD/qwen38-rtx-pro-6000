import os,sys,json,struct,time,signal
from pathlib import Path
import torch,numpy as np
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'upstream-research/src'));sys.path.insert(0,str(ROOT.parent))
from fit_flashnext_maps import load,LAYERS,sha

def capture(path):
    raw=path.read_bytes();h=struct.unpack_from('<18q',raw);n,T,cap=h[7],h[6],h[13];off=144
    def take(dtype,shape):
        nonlocal off
        count=int(np.prod(shape));dt=np.dtype(dtype);v=torch.from_numpy(np.frombuffer(raw,dtype=dt,count=count,offset=off).reshape(shape).copy());off+=count*dt.itemsize;return v
    pos=take('<i4',(n,));q=take('<f4',(n,24,256));take('<f4',(n,24,256));take('<i4',(n,cap));take('<i4',(n,4))
    k=take('i1',(T,2,256)).float();v=take('i1',(T,2,256)).float();ks=take('<f2',(T,2,4)).float();vs=take('<f2',(T,2,4)).float()
    assert off==len(raw);return pos,q,k*ks.repeat_interleave(64,-1),v*vs.repeat_interleave(64,-1)

@torch.no_grad()
def worker():
    from lsa.sglang.index.weights import read
    from lsa.core.kernels.ward import cluster_ward
    from lsa.core.kernels.triton.ward import bucket_members_from_labels
    from lsa.core.kernels.scan_select import ScanSelect
    from lsa.core.kernels.expansion import expand_in_order
    from lsa.training.native.kernels import gather_sparse
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(4096*1048576/torch.cuda.get_device_properties(0).total_memory,0)
    path=ROOT/'maps/pilot-v1/flashnext-learned-384x64.pt';maps=read(path,384,64,LAYERS)
    report={'scope':'Real learned maps and captured FlashNext inputs through CUDA index and gathered attention; standalone correctness only, no native Strata integration or quality/promotion claim','map_sha256':sha(path),'document':'test_code','device':torch.cuda.get_device_name(),'cases':[]}
    for layer in LAYERS:
        x=load('test_code',layer);d=x['d'];chosen=torch.linspace(0,len(x['rows'])-1,8).round().long();rows=x['rows'][chosen]
        wq,wk=(t.cuda() for t in maps[layer]);ik=F.normalize(d.key_inputs().cuda()@wk,dim=-1);iq=F.normalize(d.query_inputs(rows).cuda()@wq,dim=-1).bfloat16();positions=d.query_positions[rows].cuda().int().contiguous()
        assert torch.isfinite(ik).all() and torch.isfinite(iq).all()
        T=len(ik);blocks=T//512;closed=ik[:blocks*512].view(blocks,512,64);pooled=closed.view(blocks,128,4,64).sum(2)
        sizes=torch.full((blocks,128),4.,device='cuda');labels,reached=cluster_ward(pooled,sizes,64,16,precision='ieee',kernels='triton');assert reached.all()
        labels=labels.repeat_interleave(4,1);sums=torch.zeros(blocks,64,64,device='cuda');sums.scatter_add_(1,labels[...,None].expand_as(closed),closed)
        centroids=F.normalize(sums,dim=-1).reshape(-1,64).bfloat16();members=bucket_members_from_labels(labels,64,16);members=torch.where(members>=0,members+(torch.arange(blocks,device='cuda')*512)[:,None,None],-1).reshape(-1,16).int()
        flat=members[members>=0];assert torch.equal(flat.sort().values,torch.arange(blocks*512,device='cuda'))
        visible=(positions//512*64).clamp_max(len(centroids)).contiguous();select=ScanSelect(8,385,device='cuda',kernels='triton');picks=select(iq,centroids,visible).clone()
        scores=iq.float()@centroids.float().T
        for i in range(8):
            live=picks[i][picks[i]>=0].long();count=min(385,int(visible[i]));assert len(live)==count and len(live.unique())==count and bool((live<visible[i]).all())
            threshold=scores[i,:visible[i]].topk(count).values[-1];assert bool((scores[i,live]>=threshold-1e-5).all())
        expanded,lengths=expand_in_order(picks,members,positions,512,4,1024,kernels='triton')
        candidate=expanded[:,:1024];valid=(candidate>=4)&(candidate<=positions[:,None]-512)
        assert valid.all();long_ids=candidate.contiguous();mass=[]
        for i in range(8):
            held=expanded[i][expanded[i]>=0].long();assert len(held)==1540 and held.unique().numel()==1540 and int(lengths[i])==1540 and bool((held<=positions[i]).all())
            fixed=torch.cat((torch.arange(4,device='cuda'),torch.arange(int(positions[i])-511,int(positions[i])+1,device='cuda')))
            assert torch.isin(fixed,held).all();mass.append(float(x['p'][chosen[i],long_ids[i].cpu().long()].sum()))
        result={'layer':layer,'tokens':T,'query_positions':positions.cpu().tolist(),'long_range_budget':1024,'selected_per_row':1540,'selection_checks_passed':True,'conditional_native_long_range_mass':sum(mass)/len(mass)}
        if layer in (3,27,47):
            cp,q,k,v=capture(ROOT/'captures/test_code'/f'layer-{layer}-p0.lsacap');lookup={int(p):i for i,p in enumerate(cp)};qreal=torch.stack([q[lookup[int(p)]] for p in positions.cpu()])
            qfull=torch.zeros(1,T,24,256,device='cuda',dtype=torch.float16);qfull[0,positions.long()]=qreal.cuda().half();k16=k.cuda().half()[None].contiguous();v16=v.cuda().half()[None].contiguous();ids=torch.full((1,T,1024),-1,device='cuda',dtype=torch.int32);ids[0,positions.long()]=long_ids
            assert all(torch.isfinite(t).all() for t in (qfull,k16,v16))
            output,lse=gather_sparse.attention(qfull,k16,v16,ids,512,4,1/16,members=None)
            errors=[];lse_errors=[];rounding=[]
            for i,position in enumerate(positions.cpu().tolist()):
                keep=expanded[i][expanded[i]>=0].long();q64=qfull[0,position].cpu().double().reshape(2,12,256);k64=k16[0,keep].cpu().double();v64=v16[0,keep].cpu().double()
                score=torch.einsum('hgd,nhd->hgn',q64,k64)/16;want=torch.einsum('hgn,nhd->hgd',score.softmax(-1),v64).reshape(24,256);want_lse=score.logsumexp(-1).reshape(24)
                actual=output[0,position].cpu().double();err=float((actual-want).norm()/want.norm().clamp_min(1e-12));assert err<1e-2
                got_lse=lse[0,:,position].cpu().double();torch.testing.assert_close(got_lse,want_lse,atol=1e-3,rtol=1e-4)
                original_scores=torch.einsum('hgd,nhd->hgn',qreal[i].double().reshape(2,12,256),k[keep.cpu()].double())/16
                original=torch.einsum('hgn,nhd->hgd',original_scores.softmax(-1),v[keep.cpu()].double()).reshape(24,256)
                errors.append(err);lse_errors.append(float((got_lse-want_lse).abs().max()));rounding.append(float((want-original).norm()/original.norm().clamp_min(1e-12)))
            result['attention']={'query_heads':24,'kv_heads':2,'head_dim':256,'kernel_dtype':'float16','queries_checked':8,'max_relative_l2':max(errors),'max_lse_abs_error':max(lse_errors),'max_input_rounding_relative_l2':max(rounding),'passed':True}
            del qfull,k16,v16,ids,output,lse
        report['cases'].append(result);print(json.dumps(result),flush=True)
    torch.cuda.synchronize();report['status']='passed';report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1048576
    p=ROOT/'receipts/gpu-map-smoke.json';assert not p.exists();p.write_text(json.dumps(report,indent=2))

def main():
    if '--worker' in sys.argv:worker();return
    import run_research_window as guard
    def stop(sig,frame):raise KeyboardInterrupt('map smoke interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGHUP,stop)
    receipt={'started_at':guard.utc(),'script_sha256':sha(__file__)}
    with guard.kit_lock(ROOT.parent):
        receipt['preflight']=guard.check_window(4096);env=dict(os.environ);env.pop('NVIDIA_VISIBLE_DEVICES',None)
        env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(ROOT.parent/'cache/triton'),CUDA_CACHE_PATH=str(ROOT.parent/'cache/cuda'),XDG_CACHE_HOME=str(ROOT.parent/'cache/xdg'))
        try:receipt['worker']=guard.run_child([sys.executable,str(Path(__file__).resolve()),'--worker'],env=env,cwd=ROOT,log_path=ROOT/'receipts/gpu-map-smoke.log',timeout_seconds=900)
        finally:
            receipt['finished_at']=guard.utc();(ROOT/'receipts/gpu-map-smoke-window.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
        assert receipt['worker']['returncode']==0 and (ROOT/'receipts/gpu-map-smoke.json').exists()
if __name__=='__main__':main()

