"""Actual causal attention KV + MoE reference integration; tiny CPU model only."""
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from moefit.routing_replay import RoutingReplayCache
L,H,E,K=4,32,128,4

class Decoder:
    def __init__(self, model, session=None, kv=None):
        self.model,self.session=model,session
        self.kv=[(np.empty((0,H),np.float32),np.empty((0,H),np.float32)) for _ in range(L)] if kv is None else [(k.copy(),v.copy()) for k,v in kv]
        self.hits=self.routers=self.experts=0
    def fork(self):
        return Decoder(self.model,self.session.fork() if self.session else None,self.kv)
    def run(self,tokens):
        emb,qkv,gate,experts,head=self.model
        outputs=[]
        for token in tokens:
            if self.session:self.session.begin_token(token)
            x=emb[token].copy()
            for l in range(L):
                q,k,v=np.einsum('h,ahj->aj',x,qkv[l],optimize=False)
                keys,values=self.kv[l]
                keys=np.concatenate((keys,k[None,:]));values=np.concatenate((values,v[None,:]))
                self.kv[l]=(keys,values)
                scores=np.einsum('th,h->t',keys,q,optimize=False)/np.sqrt(H)
                probs=np.exp(scores-scores.max());probs/=probs.sum()
                x=np.tanh(x+np.einsum('t,th->h',probs,values,optimize=False))
                def compute():
                    self.routers+=1
                    logits=np.einsum('h,he->e',x,gate[l],optimize=False)
                    ids=np.argsort(-logits)[:K].astype(np.int16)
                    w=np.exp(logits[ids]-logits[ids].max());w/=w.sum()
                    return ids,w
                if self.session:
                    ids,w,hit=self.session.route(l,compute);self.hits+=hit
                else:ids,w=compute()
                vals=np.einsum('h,khj->kj',x,experts[l,ids],optimize=False)
                x=np.tanh(x+np.sum(vals*w[:,None],axis=0));self.experts+=1
            outputs.append(np.einsum('h,hv->v',x,head,optimize=False))
        return np.asarray(outputs)

def main():
    rng=np.random.default_rng(220)
    model=tuple(rng.normal(0,.1,shape).astype(np.float32) for shape in ((256,H),(L,3,H,H),(L,H,E),(L,E,H,H),(H,256)))
    prefix=list(range(24));suffixes=[list(range(24,48)),list(range(100,112))+list(range(36,48))]
    cache=RoutingReplayCache(compact_prefixes=True);ns=b'attention-v1/seed220/float32/zero-kv'
    root=Decoder(model,cache.layered_session(ns,L,deterministic=True));root.run(prefix)
    snapshot=[(k.copy(),v.copy()) for k,v in root.kv]
    root.fork().run(suffixes[0]);seed=cache.entries.copy();size=cache.bytes
    expected=[]
    for suffix in suffixes:
        d=Decoder(model);out=d.run(prefix+suffix)
        expected.append((out[len(prefix):],d.kv))
    def run(replay):
        cache.entries=seed.copy();cache.bytes=size
        start=time.perf_counter()
        session=cache.resume_layered(ns,L,prefix,deterministic=True) if replay else None
        parent=Decoder(model,session,snapshot);children=[];outputs=[]
        for suffix in suffixes:
            child=parent.fork();outputs.append(child.run(suffix));children.append(child)
        seconds=time.perf_counter()-start
        for i,child in enumerate(children):
            assert np.array_equal(outputs[i],expected[i][0])
            for (k,v),(ek,ev) in zip(child.kv,expected[i][1]):
                assert np.array_equal(k,ek) and np.array_equal(v,ev)
            assert child.experts==96 and child.hits==(96 if replay and i==0 else 0)
        for (k,v),(sk,sv) in zip(parent.kv,snapshot):
            assert np.array_equal(k,sk) and np.array_equal(v,sv)
            assert not np.shares_memory(k,sk)
        return dict(seconds=seconds,hits=sum(d.hits for d in children),routers=sum(d.routers for d in children),expert_layers=sum(d.experts for d in children))
    run(False);run(True)
    trials=[]
    for i in range(5):
        row={}
        for replay in ([False,True] if i%2==0 else [True,False]):row['replay' if replay else 'direct']=run(replay)
        trials.append(row)
    med={m:float(np.median([r[m]['seconds'] for r in trials])) for m in ('direct','replay')}
    result=dict(kind='MEASURED constructed causal attention+MoE CPU model, not production LLM',
        tokens=48,prefix_tokens=24,layers=L,hidden=H,experts=E,trials=trials,
        tokens_per_s={m:48/t for m,t in med.items()},speedup=med['direct']/med['replay'],hit_rate=.5,
        logits_and_all_kv_bit_exact=True,
        caveat='Actual causal attention KV, but random tiny weights, no RoPE/GQA/quantization/production kernels. Prefix execution/cache seed excluded. Direct baseline bypasses replay entirely. No deployment or SSD measurement.')
    (ROOT/'results/t7_attention_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))

if __name__=='__main__':main()
