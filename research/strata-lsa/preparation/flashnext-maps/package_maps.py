import json,hashlib,struct,sys,math
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'upstream-research/src'))
from lsa.training.distillation.data.cache import Document
from lsa.training.distillation.export import map_tau
from fit_flashnext_maps import CFG,LAYERS,sha
torch.set_num_threads(2)
manifest=json.loads((ROOT/'corpus/manifest.json').read_text());docs=manifest['documents']
assert len(docs)==8 and len({d['id'] for d in docs})==8
assert {s:sum(d['split']==s for d in docs) for s in ('train','validation','test')}=={'train':4,'validation':2,'test':2}
assert len({d['token_sha256'] for d in docs})==8
out=ROOT/'prepared-compatible';out.mkdir(exist_ok=False)
audit={'scope':'Compatibility copies only; original v1 fitting inputs retained unchanged','documents':[],'notes':['Whole-document train/validation/test override was used for v1. The original train_queries field does not define a valid within-document holdout for those maps.','Do not use train-document eval_rows to claim held-out quality; only the separate validation/test documents qualify.']}
for doc in docs:
    name=doc['id'];dest=out/name;dest.mkdir()
    assert sha(doc['token_path'])==doc['token_sha256']
    receipt=json.loads((ROOT/'captures'/name/'receipt.json').read_text());assert receipt['status']=='captured' and receipt['document']==doc
    filehash={f['name']:f['sha256'] for f in receipt['files']}
    rows=[]
    for layer in LAYERS:
        rawpath=ROOT/'captures'/name/f'layer-{layer}-p0.lsacap';raw=rawpath.read_bytes();h=struct.unpack_from('<18q',raw)
        assert h[2]==layer and h[3]==(layer-3)//4 and h[4]==0 and h[5]==h[6]==doc['tokens']
        assert hashlib.sha256(raw).hexdigest()==filehash[rawpath.name]
        nq,cap=h[7],h[13];positions=np.frombuffer(raw,dtype='<i4',count=nq,offset=144)
        assert positions.tolist()==sorted(set(range(0,h[5],32))|{h[5]-1})
        offset=144+nq*4+2*nq*24*256*4+nq*cap*4
        steps=np.frombuffer(raw,dtype='<i4',count=nq*4,offset=offset).reshape(nq,4)
        assert np.array_equal(steps[:,0],positions) and np.array_equal(steps[:,1],positions+1)
        assert np.array_equal(steps[:,2],(positions+1)//4)
        assert np.array_equal(steps[:,3],np.minimum(positions+1,2051))
        src=ROOT/'prepared'/name/f'layer-{layer}.pt';blob=torch.load(src,map_location='cpu',weights_only=True)
        assert blob['meta']['split']==doc['split'] and blob['meta']['layer']==layer and blob['meta']['capture_sha256']==filehash[rawpath.name]
        old=Document.from_blob(src,blob,CFG);rr=torch.arange(len(old.query_positions));native=old.key_target(rr,target='native_attention')
        blob={**blob,'meta':{**blob['meta'],'split_contract':'whole_document','row_holdout_valid':False if doc['split']=='train' else True,'source_prepared_sha256':sha(src)}}
        bad=blob['top_key_ids']<0;assert bool((blob['top_key_mass'][bad]==0).all())
        blob['top_key_ids']=blob['top_key_ids'].clamp_min(0)
        dst=dest/src.name;torch.save(blob,dst);d=Document.from_blob(dst,blob,CFG)
        torch.testing.assert_close(d.key_target(rr,target='native_attention'),native,atol=0,rtol=0)
        for target in ('dense','dense_on_native'):
            value=d.key_target(rr,target=target);assert torch.isfinite(value).all()
            torch.testing.assert_close(value.sum(-1),torch.ones(len(rr)),rtol=1e-5,atol=1e-5)
        rows.append({'layer':layer,'capture_sha256':filehash[rawpath.name],'original_prepared_sha256':sha(src),'compatible_prepared_sha256':sha(dst),'native_target_unchanged':True,'dense_targets_valid':True,'identity_sampling_steps_verified':True})
    audit['documents'].append({'id':name,'split':doc['split'],'layers':rows})
    print(json.dumps({'event':'compatibility_checked','document':name,'layers':12}),flush=True)
run=ROOT/'maps/pilot-v1';report=json.loads((run/'report.json').read_text());audit['map_sidecars']={}
for arm in ('pca','learned'):
    p=run/f'flashnext-{arm}-384x64.pt';tau={str(l):float(report['layers'][str(l)]['training'][f'{arm}_tau']) for l in LAYERS}
    assert all(math.isfinite(t) and t>0 for t in tau.values())
    side={'input':'key_and_query','query_heads':'all','tau':tau,'sha256':sha(p),'run':str(run),'source_report_sha256':sha(run/'report.json'),'scope':report['scope'],'input_precision':report['parameters']['input_precision']}
    sidepath=p.with_suffix('.json');assert not sidepath.exists();sidepath.write_text(json.dumps(side,indent=2))
    assert map_tau(p,LAYERS)=={int(k):v for k,v in tau.items()}
    audit['map_sidecars'][arm]={'path':str(sidepath),'sha256':sha(sidepath),'upstream_temperature_roundtrip':True}
audit['status']='passed';audit['original_report_sha256']=sha(run/'report.json')
(ROOT/'receipts/compatibility-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({'event':'finished','status':'passed','captures':96,'compatible_blobs':96}),flush=True)
