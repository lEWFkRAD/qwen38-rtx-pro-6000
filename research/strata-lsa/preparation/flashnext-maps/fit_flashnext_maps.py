import argparse,copy,hashlib,json,math,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'upstream-research/src'))
from lsa.sglang.settings.design import IndexerConfig
from lsa.sglang.index.weights import read as read_maps
from lsa.training.distillation.data.cache import Document
from lsa.training.distillation.model import IndexMap
from lsa.core import reference as reader
LAYERS=tuple(range(3,48,4));BUDGETS=(512,1024)
CFG=IndexerConfig(input='key_and_query',query_heads='all',index_dim=64,window_size=512,sink_size=4,pool_size=4,bucket_size=8,bucket_cap=16,top_k=1024,top_p=385)
torch.set_num_threads(2)
def emit(v):print(json.dumps(v),flush=True)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(name,layer):
    path=ROOT/'prepared'/name/f'layer-{layer}.pt';blob=torch.load(path,map_location='cpu',weights_only=True)
    d=Document.from_blob(path,blob,CFG)
    assert d.layer==layer and d.model_dim==384 and d.head_count==24 and d.device.type=='cpu'
    assert d.native_key_ids is not None and d.native_key_mass is not None
    assert d.block_size is None and d.closed_block is None
    assert d.summed_query.shape==(len(d.query_positions),384)
    assert torch.isfinite(d.key_inputs()).all() and torch.isfinite(d.summed_query).all()
    torch.testing.assert_close(d.native_key_mass.float().sum(-1),torch.full((len(d.query_positions),),24.),rtol=0,atol=2e-4)
    rows=torch.arange(len(d.query_positions));visible=d.visible_keys(rows);raw=d.native_attention(rows).float()/24
    keep=(visible.sum(-1)>=1024)&((raw*visible).sum(-1)>1e-9);rows=rows[keep]
    assert len(rows)>0
    target=d.key_target(rows,target='native_attention').float()
    torch.testing.assert_close(target.sum(-1),torch.ones(len(rows)),rtol=0,atol=2e-6)
    return dict(d=d,rows=rows,visible=visible[keep],p=target,raw=raw[keep],path=str(path),sha256=sha(path))
def scores(m,x,batch=None):
    rows=x['rows'] if batch is None else x['rows'][batch]
    return m.index_queries(x['d'].query_inputs(rows))@m.index_keys(x['d'].key_inputs()).T
def kl(s,p,visible,tau):
    logq=torch.log_softmax((s/tau).masked_fill(~visible,-torch.inf),-1).masked_fill(~visible,0)
    return (p*(p.clamp_min(1e-30).log()-logq)).sum(-1)
@torch.no_grad()
def monitor(m,data):return sum(float(kl(scores(m,x),x['p'],x['visible'],m.log_tau.exp()).mean()) for x in data)/len(data)
def fit(data,layer):
    torch.manual_seed(1701+layer);gen=torch.Generator().manual_seed(1701+layer)
    init=torch.cat([x['d'].key_inputs()[::max(1,x['d'].key_count//1024)].float() for x in data])
    pca=IndexMap(384,64,init=init,bases=None).cpu()
    with torch.no_grad():
        held=[scores(pca,x) for x in data];grid=torch.logspace(-3,0,25)
        losses=[sum(float(kl(s,x['p'],x['visible'],tau).mean()) for s,x in zip(held,data))/len(data) for tau in grid]
        pca.set_tau(float(grid[min(range(len(grid)),key=losses.__getitem__)]))
    model=copy.deepcopy(pca)
    opt=torch.optim.AdamW([{'params':[model.query_projection,model.key_projection],'lr':3e-4},{'params':[model.log_tau],'lr':1e-2}],weight_decay=0)
    history={0:monitor(model,data)};limit=300;step=0
    while step<limit:
        x=data[step%len(data)];batch=torch.randint(len(x['rows']),(16,),generator=gen)
        opt.zero_grad(set_to_none=True);loss=kl(scores(model,x,batch),x['p'][batch],x['visible'][batch],model.log_tau.exp()).mean()
        assert torch.isfinite(loss),'nonfinite training loss'
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step()
        with torch.no_grad():model.log_tau.clamp_(math.log(1e-3),math.log(1.0))
        step+=1
        if step%100==0:
            history[step]=monitor(model,data);emit({'event':'training','layer':layer,'step':step,'train_kl':history[step]})
        if step==300 and history[300]<history[200]*0.995:limit=600
    return {'pca':pca,'learned':model},{'steps':step,'train_kl':history,'pca_tau':pca.tau,'learned_tau':model.tau,'continuation':'300 additional steps only if train KL drops more than0.5percent from200to300'}
def selection(ids,x,budget):
    ids=ids.long();n=x['d'].key_count
    assert bool(((ids>=-1)&(ids<n)).all())
    columns=torch.where(ids>=0,ids,n);held=torch.zeros(len(ids),n+1,dtype=torch.bool);held.scatter_(1,columns,True);held=held[:,:n]
    assert bool((held.sum(-1)==(ids>=0).sum(-1)).all()),'duplicate selected IDs'
    long=held&x['visible'];count=long.sum(-1)
    if not bool((count==budget).all()):raise RuntimeError(f'Budget mismatch {budget}: {count.min()}..{count.max()}')
    return {'long_range_keys':budget,'conditional_long_range_mass':float((long*x['p']).sum(-1).mean()),'absolute_long_range_mass':float((long*x['raw']).sum(-1).mean()),'total_native_mass_retained':float((held*x['raw']).sum(-1).mean())}
@torch.no_grad()
def evaluate(model,x):
    d=x['d'];keys=model.index_keys(d.key_inputs());queries=model.index_queries(d.query_inputs(x['rows']));s=queries@keys.T
    buckets=reader.build_buckets(keys,4,8,16,512)
    ranking=torch.argsort(s.masked_fill(~x['visible'],-torch.inf),dim=-1,descending=True,stable=True)
    report={'kl':float(kl(s,x['p'],x['visible'],model.log_tau.exp()).mean()),'query_rows_used':len(x['rows']),'query_rows_total':len(d.query_positions),'budgets':{}}
    for budget in BUDGETS:
        ids,lengths=reader.read(queries,d.query_positions[x['rows']],buckets,385,budget,4,512)
        result=selection(ids,x,budget);result['ideal_per_key_mass']=float(x['p'].gather(1,ranking[:,:budget]).sum(-1).mean());result['teacher_oracle_mass']=float(x['p'].topk(budget,dim=-1).values.sum(-1).mean())
        report['budgets'][str(budget)]=result
    return report

def summarize(report,split):
    paired=[]
    for layer,row in report['layers'].items():
        for name,result in row.get(split,{}).items():paired.append((layer,name,result))
    if not paired:return None
    out={'pairs':len(paired),'mean_kl':{},'mean_conditional_mass':{},'worst_mass_delta':{}}
    for arm in ('pca','learned'):
        out['mean_kl'][arm]=sum(r[arm]['kl'] for _,_,r in paired)/len(paired)
        out['mean_conditional_mass'][arm]={str(b):sum(r[arm]['budgets'][str(b)]['conditional_long_range_mass'] for _,_,r in paired)/len(paired) for b in BUDGETS}
    for b in BUDGETS:
        worst=min((r['learned']['budgets'][str(b)]['conditional_long_range_mass']-r['pca']['budgets'][str(b)]['conditional_long_range_mass'],l,n) for l,n,r in paired)
        out['worst_mass_delta'][str(b)]={'delta':worst[0],'layer':worst[1],'document':worst[2]}
    out['pilot_gate_passed']=out['mean_kl']['learned']<out['mean_kl']['pca'] and all(out['mean_conditional_mass']['learned'][str(b)]>=out['mean_conditional_mass']['pca'][str(b)] and out['worst_mass_delta'][str(b)]['delta']>=-0.03 for b in BUDGETS)
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='pilot-v1');args=parser.parse_args()
    out=ROOT/'maps'/args.output;out.mkdir(exist_ok=False)
    manifest=json.loads((ROOT/'corpus/manifest.json').read_text());names={s:sorted(d['id'] for d in manifest['documents'] if d['split']==s) for s in ('train','validation','test')}
    report={'schema_version':1,'scope':'Flash-Next IQ3_S INT8-cache pilot maps; no model-quality/serving-promotion claim','created_unix':time.time(),'dataset_sha256':sha(ROOT/'corpus/manifest.json'),'capture_plan_sha256':sha(ROOT/'capture-plan.json'),'source':'LSA906be195 plus CPU Ward symmetry repair35b1bf9','parameters':{'input_dim':384,'index_dim':64,'layers':list(LAYERS),'budget':list(BUDGETS),'top_p':385,'window':512,'sink':4,'batch':16,'optimizer':'AdamW weight_decay0','matrix_lr':3e-4,'temperature_lr':1e-2,'clip':1.,'input_precision':'FP32 dequantized INT8 K; FP32 Q group sums','pilot_gate':'mean KL improves; mean retained mass at512and1024doesnotdecrease; no layer-document mass regression larger than0.03'},'splits':names,'layers':{}}
    models={};exports={'pca':{},'learned':{}}
    for layer in LAYERS:
        emit({'event':'layer_start','layer':layer});data=[load(name,layer) for name in names['train']]
        arms,training=fit(data,layer);models[layer]=arms
        report['layers'][str(layer)]={'training':training,'train_inputs':[{'path':x['path'],'sha256':x['sha256']} for x in data]}
        for arm,m in arms.items():
            wq,wk=m.served_projections()
            for kind,tensor in (('W_Q_idx',wq),('W_K_idx',wk)):
                value=tensor.detach().cpu().float().contiguous().clone();assert value.shape==(384,64) and torch.isfinite(value).all()
                exports[arm][f'{layer}.{kind}']=value
        torch.save({arm:m.state_dict() for arm,m in arms.items()},out/f'layer-{layer}.checkpoint.pt')
        (out/'progress.json').write_text(json.dumps(report,indent=2))
    for arm,flat in exports.items():
        assert len(flat)==24;path=out/f'flashnext-{arm}-384x64.pt';torch.save(flat,path);loaded=read_maps(path,384,64,LAYERS)
        for layer in LAYERS:
            wq,wk=models[layer][arm].served_projections();assert torch.equal(loaded[layer][0],wq.detach()) and torch.equal(loaded[layer][1],wk.detach())
        report.setdefault('exports',{})[arm]={'path':str(path),'sha256':sha(path),'roundtrip_exact':True,'tensors':24,'parameters':589824}
    emit({'event':'all_candidates_frozen','layers':12,'exported_maps':2})
    # No validation/test inputs were used for optimization or continuation.
    for split in ('validation','test'):
        for layer in LAYERS:
            row=report['layers'][str(layer)];row[split]={}
            for name in names[split]:
                x=load(name,layer);row[split][name]={arm:evaluate(model,x) for arm,model in models[layer].items()}
            emit({'event':'evaluated','split':split,'layer':layer})
            (out/'progress.json').write_text(json.dumps(report,indent=2))
        report.setdefault('summary',{})[split]=summarize(report,split);emit({'event':'split_summary','split':split,'result':report['summary'][split]})
    report['status']='pilot_qualified' if all(x['pilot_gate_passed'] for x in report['summary'].values()) else 'maps_trained_pilot_gate_not_met'
    report['production_promotion']=False
    (out/'report.json').write_text(json.dumps(report,indent=2));emit({'event':'finished','status':report['status'],'report':str(out/'report.json')})
if __name__=='__main__':main()
