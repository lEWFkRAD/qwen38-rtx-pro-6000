import json,torch
import torch.nn.functional as F
from lsa.core.kernels.ward import cluster_ward
from lsa.core.kernels.triton.ward import bucket_members_from_labels
from lsa.core.kernels.scan_select import ScanSelect
from lsa.core.kernels.expansion import expand_in_order
from lsa.training.native.kernels import gather_sparse
from tests.helpers.training.kernels.inputs import dense_reference,keep_table,relative_error
BLOCK,POOL,BUCKET,CAP=512,4,8,16
DIM,TOP_P,TOP_K,WINDOW,SINK=64,64,512,512,4
torch.cuda.set_per_process_memory_fraction(4096*1048576/torch.cuda.get_device_properties(0).total_memory,0)
torch.backends.cuda.matmul.allow_tf32=False

def case(tokens,heads):
    torch.cuda.reset_peak_memory_stats()
    gen=torch.Generator(device='cuda').manual_seed(1729)
    def draw(*shape):return torch.randn(*shape,generator=gen,device='cuda',dtype=torch.bfloat16).requires_grad_()
    q,k,v=draw(1,tokens,heads,DIM),draw(1,tokens,1,DIM),draw(1,tokens,1,DIM)
    with torch.no_grad():
        wq=torch.randn(DIM,DIM,generator=gen,device='cuda')
        wk=torch.randn(DIM,DIM,generator=gen,device='cuda')
        iq=F.normalize(q[0].float().sum(1)@wq,dim=-1).bfloat16()
        ik=F.normalize(k[0,:,0].float()@wk,dim=-1)
        blocks=tokens//BLOCK; per=BLOCK//BUCKET
        closed=ik[:blocks*BLOCK].view(blocks,BLOCK,DIM)
        pooled=closed.view(blocks,BLOCK//POOL,POOL,DIM).sum(2)
        sizes=torch.full((blocks,BLOCK//POOL),float(POOL),device='cuda')
        labels,reached=cluster_ward(pooled,sizes,per,CAP,precision='ieee',kernels='triton')
        assert bool(reached.all())
        labels=labels.repeat_interleave(POOL,dim=1)
        sums=torch.zeros(blocks,per,DIM,device='cuda')
        sums.scatter_add_(1,labels[...,None].expand_as(closed),closed)
        centroids=F.normalize(sums,dim=-1).reshape(-1,DIM).bfloat16()
        members=bucket_members_from_labels(labels,per,CAP)
        offsets=(torch.arange(blocks,device='cuda')*BLOCK)[:,None,None]
        members=torch.where(members>=0,members+offsets,-1).reshape(-1,CAP).int()
        assert int(members.max())<blocks*BLOCK
        positions=torch.arange(tokens,device='cuda',dtype=torch.int32)
        visible=(positions//BLOCK*per).clamp_max(blocks*per).contiguous()
        picks=torch.empty(tokens,TOP_P,device='cuda',dtype=torch.int32)
        select=ScanSelect(256,TOP_P,device='cuda',kernels='triton')
        for start in range(0,tokens,256):
            stop=min(start+256,tokens)
            picks[start:stop]=select(iq[start:stop],centroids,visible[start:stop])
        expanded,lengths=expand_in_order(picks,members,positions,WINDOW,SINK,TOP_K,kernels='triton')
        candidate=expanded[:,:TOP_K]
        long=(candidate>=SINK)&(candidate<=positions[:,None]-WINDOW)
        # Pack long-range IDs before padding: reader bounds each row's prefix by its possible old keys.
        order=torch.argsort(long.int(),dim=1,descending=True,stable=True)
        ordered=candidate.gather(1,order); valid=long.gather(1,order)
        ids=torch.where(valid,ordered,-1)[None].contiguous()
        assert bool((ids[0,:WINDOW+SINK]==-1).all())
        assert bool((ids>=0).any())
        pos=torch.arange(tokens,device='cuda'); delta=pos[:,None]-pos[None,:]
        near=(delta>=0)&((delta<WINDOW)|(pos[None,:]<SINK))
        expected_keep=keep_table(expanded[None],tokens)
        reader_keep=keep_table(ids,tokens)|near[None]
        assert torch.equal(expected_keep,reader_keep),'Adapter changed selected keep set'
        assert torch.equal(lengths,expected_keep[0].sum(-1).int())
        for row in ids[0].cpu().tolist():
            live=[x for x in row if x>=0]
            assert len(live)==len(set(live))
        del expected_keep,reader_keep,near,delta
    output,lse=gather_sparse.attention(q,k,v,ids,WINDOW,SINK,DIM**-0.5)
    grad=torch.randn(output.shape,generator=gen,device='cuda',dtype=output.dtype)
    got=torch.autograd.grad(output,(q,k,v),grad)
    leaves=[x.detach().float().requires_grad_() for x in (q,k,v)]
    want,want_lse=dense_reference(*leaves,ids,WINDOW,SINK,DIM**-0.5)
    expected_grads=torch.autograd.grad(want,leaves,grad.float())
    errors={'output':relative_error(output,want),**{'d'+name:relative_error(a,b) for name,a,b in zip('qkv',got,expected_grads)}}
    assert all(e<1e-2 for e in errors.values()),errors
    torch.testing.assert_close(lse,want_lse,atol=1e-3,rtol=1e-4)
    torch.cuda.synchronize()
    return dict(status='passed',tokens=tokens,query_heads=heads,kv_heads=1,head_dim=DIM,top_k=TOP_K,partial_final_block=tokens%BLOCK,selected_long_range=int((ids>=0).sum()),relative_errors=errors,lse_max_abs_error=float((lse-want_lse).abs().max()),peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,scope='Synthetic fixed-map selection to sparse read and QKV gradients; no model quality or Strata integration claim')
for tokens,heads in ((4096,1),(4103,1),(4096,4)):
    print(json.dumps(case(tokens,heads)),flush=True)
