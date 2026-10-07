import sys, json, numpy as np
sys.path.insert(0,'experiments')
import design_xlayer_guarded as g
W,tags,X,I=g.load(sys.argv[1])
P2=[g.preview(W,x,2) for x in X]
out=[]
for k in range(len(I)):
    # each prompt alone, cold cache, warm=32 tokens unscored
    b=g.simulate([P2[k]],[I[k]],143,24.1,mode="baseline",warm=32)
    d=g.simulate([P2[k]],[I[k]],143,24.1,issue="at_xL",d=2,tau=0.01,staged=True,warm=32)
    out.append((tags[k],b['tps'],d['tps'],round(d['tps']/b['tps']-1,4),b['miss_per_tok'],d['miss_per_tok']))
    print(out[-1],flush=True)
r=[o[3] for o in out]; print('per-prompt gain min/median/max',min(r),float(np.median(r)),max(r))
json.dump(out,open('results/design_xlayer_guarded_perprompt.json','w'),indent=1)
