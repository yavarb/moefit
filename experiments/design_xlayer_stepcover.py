import sys, numpy as np
sys.path.insert(0,'experiments')
import design_xlayer_guarded as g
from collections import OrderedDict
W,tags,X,I=g.load(sys.argv[1])
P2=[g.preview(W,x,2) for x in X]
cap=143; caches=[OrderedDict() for _ in range(48)]
steps_with_miss=full=part=0; nm=cov=0; tok=0
for P,Ii in zip(P2,I):
    for t in range(Ii.shape[0]):
        for L in range(48):
            c=caches[L]; miss=[int(e) for e in Ii[t,L] if int(e) not in c]
            if L>=2 and tok>=160 and miss:
                pv=P[t,L]; o=[int(e) for e in np.argsort(-pv)[:40] if pv[e]>=0.01 and int(e) not in c][:10]
                k=sum(e in o for e in miss); steps_with_miss+=1; nm+=len(miss); cov+=k
                full+= k==len(miss); part+= 0<k<len(miss)
            for e in Ii[t,L]:
                e=int(e)
                if e in c: c.move_to_end(e)
                else:
                    c[e]=1
                    if len(c)>cap: c.popitem(last=False)
        tok+=1
print(f"layer-steps with misses: {steps_with_miss}; mean misses/step {nm/steps_with_miss:.2f}; miss coverage {cov/nm:.3f}")
print(f"ALL misses covered: {full/steps_with_miss:.3f}  partial: {part/steps_with_miss:.3f}  none: {1-(full+part)/steps_with_miss:.3f}")
