"""CPU-only diagnostics of unmodified upstream reference failures."""
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NVIDIA_VISIBLE_DEVICES'] = 'void'
import torch
from lsa.core.ward import pool_keys, ward_rounds
from lsa.core.attention import topk_attention
import torch.nn.functional as F

torch.set_num_threads(2)
root = Path(__file__).resolve().parent

def trace(keys, sizes, count, cap):
    # Instrumented copy of the merge loop; upstream source remains untouched.
    groups, pools, _ = keys.shape
    sums, sizes = keys.clone(), sizes.clone()
    alive = torch.ones(groups, pools, dtype=torch.bool)
    index = torch.arange(pools).expand(groups, pools)
    diagonal = torch.eye(pools, dtype=torch.bool)
    max_asym = 0.0
    for iteration in range(256):
        left = alive.sum(1) - count
        if not (left > 0).any():
            return {'reached':True, 'rounds':iteration, 'max_gram_asymmetry':max_asym}
        gram = torch.bmm(sums, sums.transpose(1,2))
        max_asym = max(max_asym, float((gram-gram.transpose(1,2)).abs().max()))
        squares=gram.diagonal(dim1=1,dim2=2)
        merged=(squares[:,:,None]+squares[:,None,:]+2*gram).clamp_min(0).sqrt()
        norms=squares.clamp_min(0).sqrt()
        costs=norms[:,:,None]+norms[:,None,:]-merged
        allowed=alive[:,:,None]&alive[:,None,:]&~diagonal
        allowed &= sizes[:,:,None]+sizes[:,None,:] <= cap
        minimum,best=costs.masked_fill(~allowed,float('inf')).min(2)
        mutual=(minimum<float('inf'))&(best.gather(1,best)==index)&(index<best)
        rank=minimum.masked_fill(~mutual,float('inf')).argsort(1).argsort(1)
        mutual &= rank<left.clamp_min(0)[:,None]
        if not mutual.any():
            blocked=(left>0).nonzero().flatten()
            return {'reached':False,'rounds':iteration,'remaining_counts':alive.sum(1).tolist(),'target':count,'blocked_groups':blocked.tolist(),'allowed_pairs_in_blocked_groups':[int(allowed[g].sum()) for g in blocked],'cost_asymmetry_at_stop':float((costs-costs.transpose(1,2)).abs().max()),'max_gram_asymmetry':max_asym}
        group, kept=mutual.nonzero(as_tuple=True)
        absorbed=best[group,kept]
        sums[group,kept]+=sums[group,absorbed]
        sizes[group,kept]+=sizes[group,absorbed]
        alive[group,absorbed]=False
    return {'reached':False,'reason':'max_rounds'}

out={'scope':'CPU diagnostics; no upstream patch and no GPU qualification','ward':[],'attention':[]}
for pool in (1,4):
    for seed in range(6):
        gen=torch.Generator().manual_seed(seed)
        keys=F.normalize(torch.randn(8,128*pool,16,generator=gen),dim=-1)
        sums=pool_keys(keys,pool)
        sizes=torch.full((8,128),float(pool))
        if seed%3==1:
            sums=sums[:,:1].expand_as(sums)+1e-3*sums
        if seed%3==2:
            sums=sums.round()
        _, reached=ward_rounds(sums,sizes,128*pool//8,16)
        record={'pool':pool,'seed':seed,'upstream_fp32_reached':reached.tolist()}
        if not reached.all():
            record['fp32_trace']=trace(sums,sizes,128*pool//8,16)
            _, double_reached=ward_rounds(sums.double(),sizes.double(),128*pool//8,16)
            record['diagnostic_fp64_reached']=double_reached.tolist()
        out['ward'].append(record)
for seed in range(4):
    torch.manual_seed(seed)
    b,t,hq,hkv,dh=2,96,8,2,16
    q,k,v=torch.randn(b,t,hq,dh),torch.randn(b,t,hkv,dh),torch.randn(b,t,hkv,dh)
    selected=torch.arange(t,dtype=torch.int32).expand(b,t,t).contiguous()
    full=topk_attention(q,k,v,selected,window=0,sink=0,scale=dh**-0.5,query_block_size=t)
    split=topk_attention(q,k,v,selected,window=0,sink=0,scale=dh**-0.5,query_block_size=7)
    out['attention'].append({'seed':seed,'bitwise_equal':torch.equal(full,split),'max_abs_error':float((full-split).abs().max()),'allclose_atol_1e-6_rtol_1e-5':torch.allclose(full,split,atol=1e-6,rtol=1e-5),'finite':bool(torch.isfinite(full).all() and torch.isfinite(split).all())})
path=root/'receipts/reference-diagnostics.json'
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
