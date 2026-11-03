import argparse,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'upstream-research/src'))
from lsa.training.distillation.data.cache import Document
from lsa.sglang.settings.design import IndexerConfig
CFG=IndexerConfig(input='key_and_query',query_heads='all',index_dim=64,sink_size=4,window_size=512,top_k=1024,top_p=385)
torch.set_num_threads(2)
def prepare(path,split):
    raw=path.read_bytes();h=struct.unpack_from('<18q',raw)
    magic,version,layer,ordinal,p0,T,keys,nq=h[:8];page,cap=h[12:14]
    assert magic==0x4c53414d41503031 and version==1 and h[8:12]==(24,2,256,64) and h[14:]==(4,32,0,0)
    assert p0==0 and keys==T and 1024<=T<=8192 and 1<=nq<=258 and 1<=cap<=8192
    off=144
    def take(dtype,shape):
        nonlocal off
        dt=np.dtype(dtype);n=int(np.prod(shape));end=off+n*dt.itemsize
        assert end<=len(raw),'truncated capture'
        value=torch.from_numpy(np.frombuffer(raw,dtype=dt,count=n,offset=off).reshape(shape).copy());off=end;return value
    pos=take('<i4',(nq,)).long();q=take('<f4',(nq,24,256));actual=take('<f4',(nq,24,256))
    ids=take('<i4',(nq,cap)).long();steps=take('<i4',(nq,4)).long()
    kc=take('i1',(keys,2,256));vc=take('i1',(keys,2,256));ks=take('<f2',(keys,2,4)).float();vs=take('<f2',(keys,2,4)).float()
    assert off==len(raw) and torch.equal(steps[:,0],pos) and torch.equal(steps[:,1],pos+1)
    assert bool((pos[1:]>pos[:-1]).all()) and bool(((pos>=0)&(pos<keys)).all())
    assert all(bool(torch.isfinite(x).all()) for x in (q,actual,ks,vs)) and bool((ks>=0).all() and (vs>=0).all())
    k=kc.float()*ks.repeat_interleave(64,-1);v=vc.float()*vs.repeat_interleave(64,-1)
    ni=[];nm=[];ti=[];tm=[];peaks=[];diagnostics=[];selected=[];allkeys=torch.arange(keys)
    for row,position in enumerate(pos.tolist()):
        width=int(steps[row,3]);assert 1<=width<=cap
        held=ids[row,:width];assert bool(((held>=0)&(held<=position)).all()) and held.unique().numel()==width
        query=q[row].double().reshape(2,12,256)
        score=torch.einsum('hgd,nhd->hgn',query,k[held].double())/16
        probability=score.softmax(-1)
        oracle=torch.einsum('hgn,nhd->hgd',probability,v[held].double()).reshape(24,256)
        delta=actual[row].double()-oracle
        scale=max(float(oracle.abs().max()),float(actual[row].abs().max()),1e-12)
        relative=float(delta.abs().max())/scale
        diagnostics.append({'position':position,'width':width,'relative_max_scale':relative,'relative_l2':float(delta.norm())/max(float(oracle.norm()),1e-12)})
        if position<1024:continue
        selected.append(row)
        row_ids=torch.full((cap,),-1,dtype=torch.int32);mass=torch.zeros(cap)
        row_ids[:width]=held.int();mass[:width]=probability.sum((0,1)).float();ni.append(row_ids);nm.append(mass)
        visible=allkeys[(allkeys>=4)&(position-allkeys>=512)]
        dense=torch.einsum('hgd,nhd->hgn',query,k[visible].double())/16
        dense_mass=dense.softmax(-1).sum((0,1));count=min(2048,len(visible));values,order=dense_mass.topk(count)
        row_ids=torch.full((2048,),-1,dtype=torch.int32);mass=torch.zeros(2048)
        row_ids[:count]=visible[order].int();mass[:count]=values.float();ti.append(row_ids);tm.append(mass)
        peaks.append(visible[dense.argmax(-1)].reshape(24).int())
    report={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'layer':layer,'ordinal':ordinal,'tokens':keys,'queries':nq,'eligible_queries':len(selected),'oracle_tolerance':1e-4,'oracle_rows':diagnostics,'max_relative_max_scale':max(d['relative_max_scale'] for d in diagnostics),'max_relative_l2':max(d['relative_l2'] for d in diagnostics),'native_mass_convention':'sum_of_24_independent_head_probabilities'}
    report['passed']=report['max_relative_max_scale']<=1e-4
    if not report['passed']:return None,report
    chosen=torch.tensor(selected);assert len(chosen)>1
    blob={'key':k[:,:,64:].reshape(keys,384).contiguous(),'summed_query':q[chosen].reshape(-1,2,12,256)[:,:,:,64:].sum(2).reshape(-1,384),'query_positions':pos[chosen],'native_key_ids':torch.stack(ni),'native_key_mass':torch.stack(nm),'top_key_ids':torch.stack(ti),'top_key_mass':torch.stack(tm),'peak_key_id':torch.stack(peaks),'meta':{'model':'retained-flash-next-iq3_s-int8kv-control-vector','layer':layer,'split':split,'sink_size':4,'window_size':512,'cluster_block_size':512,'train_queries':len(chosen)-1 if split=='train' else 0,'capture_sha256':report['sha256'],'input_precision':'FP32 dequantized INT8 K; FP32 grouped Q; normalize only after projection'}}
    document=Document.from_blob(path,blob,CFG);target=document.key_target(torch.arange(len(chosen)),target='native_attention')
    assert torch.isfinite(target).all() and torch.allclose(target.sum(-1),torch.ones(len(chosen)),atol=1e-5,rtol=1e-5)
    # Independently check grouping on the same nonrotary coordinates.
    pick=min(17,keys-1);query=q[chosen[0],:,64:].double().reshape(2,12,192)
    lhs=(blob['summed_query'][0].double()*blob['key'][pick].double()).sum()
    rhs=(query*k[pick,:,64:].double()[:,None,:]).sum()
    assert torch.allclose(lhs,rhs,atol=1e-3,rtol=1e-5)
    report['grouped_dot_verified']=True
    return blob,report

def main():
    p=argparse.ArgumentParser();p.add_argument('--documents',nargs='+',required=True);a=p.parse_args()
    manifest=json.loads((ROOT/'corpus/manifest.json').read_text());docs={d['id']:d for d in manifest['documents']}
    for name in a.documents:
        doc=docs[name];out=ROOT/'prepared'/name;out.mkdir(parents=True,exist_ok=False)
        reports=[]
        for layer in range(3,48,4):
            path=ROOT/'captures'/name/f'layer-{layer}-p0.lsacap'
            blob,report=prepare(path,doc['split']);reports.append(report)
            (out/f'layer-{layer}.json').write_text(json.dumps(report,indent=2))
            if blob is None:raise RuntimeError(f'Native replay mismatch {name} layer{layer}: {report["max_relative_max_scale"]}')
            torch.save(blob,out/f'layer-{layer}.pt')
            print(json.dumps({'document':name,'layer':layer,'native_replay_relative_max':report['max_relative_max_scale'],'queries':report['eligible_queries'],'status':'prepared'}),flush=True)
        (out/'summary.json').write_text(json.dumps({'document':doc,'layers':12,'max_replay_relative_max':max(r['max_relative_max_scale'] for r in reports),'passed':True},indent=2))
if __name__=='__main__':main()
