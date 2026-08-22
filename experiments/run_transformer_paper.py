"""Compute Transformer Filter numbers in the EXACT three harnesses used for the
paper tables (run_pureattack / run_matched / run_tableB), so the new Transformer
rows are directly consistent with the existing FCN/LSTM/Signal/naive numbers.
Only the Transformer is evaluated (fast)."""
import sys, os, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque
from torchvision import datasets, transforms
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_switching as E
CKPT=os.path.dirname(os.path.abspath(__file__)); W=10; L=50
device=torch.device('cuda')
pool=E.load_pool('divergent_kd_v5b_pool',device); K=len(pool)
class Tr(nn.Module):
    def __init__(s,d=32,nhead=4,nl=2):
        super().__init__(); s.emb=nn.Linear(1,d)
        pe=torch.zeros(L,d); pos=torch.arange(L).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d,2).float()*(-np.log(10000.0)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div); s.register_buffer('pe',pe)
        el=nn.TransformerEncoderLayer(d_model=d,nhead=nhead,dim_feedforward=64,batch_first=True,dropout=0.0)
        s.enc=nn.TransformerEncoder(el,nl); s.fc=nn.Linear(d,1)
    def forward(s,x):
        h=s.emb(x.unsqueeze(-1))+s.pe.unsqueeze(0); h=s.enc(h); return s.fc(h.mean(1)).squeeze(1)
trf=Tr(); trf.load_state_dict(torch.load(os.path.join(CKPT,'dl_transformer.pth'),map_location='cpu')); trf.eval()
def confirm(w):
    with torch.no_grad(): return torch.sigmoid(trf(torch.tensor(w,dtype=torch.float32).unsqueeze(0))).item()

class Oracle:  # transformer gate-confirm + greedy selection
    def __init__(s,thr,rng): s.thr=thr; s.rng=rng
    def reset(s,start=0): s.cur=start; s.query_count=0; s.switches=0; s.hist=deque(maxlen=L)
    def __call__(s,x):
        if isinstance(x,np.ndarray): x=torch.from_numpy(x).float()
        xt=x.to(device)
        with torch.no_grad(): sc=F.softmax(pool[s.cur](xt),dim=1).cpu().numpy()[0]
        ss=np.sort(sc)[::-1]; s.hist.append(ss[0]-ss[1]); s.query_count+=1; h=list(s.hist); fire=False
        if len(h)>=W and np.mean(h[-W:])<0.6:
            w=np.array(h[-L:])
            if len(w)<L: w=np.pad(w,(L-len(w),0),constant_values=w[0])
            fire=confirm(w)>s.thr
        if fire:
            others=[i for i in range(K) if i!=s.cur]; ms=[]
            with torch.no_grad():
                for m in pool:
                    c=F.softmax(m(xt),dim=1).cpu().numpy()[0]; cs=np.sort(c)[::-1]; ms.append(cs[0]-cs[1])
            s.cur=max(others,key=lambda i:ms[i]); s.switches+=1; s.hist.clear()
        return sc[None]

tf=transforms.Compose([transforms.ToTensor()]); testset=datasets.CIFAR100(root=E.DATA_ROOT,train=False,download=False,transform=tf)
loader=torch.utils.data.DataLoader(testset,batch_size=256,shuffle=False,num_workers=2); correct,off=[],0
with torch.no_grad():
    for im,la in loader:
        im,la=im.to(device),la.to(device); bc=torch.ones(la.size(0),dtype=torch.bool,device=device)
        for m in pool: bc&=(m(im).argmax(1)==la)
        for j in torch.nonzero(bc).squeeze(1).tolist(): correct.append(off+j)
        off+=la.size(0)
correct=np.array(correct)
out={}

# ---- Setting 1: run_pureattack replica (idxs from RandomState(7), seeds 4000+k) ----
glob=np.random.RandomState(7); idxs=glob.choice(correct,200,replace=False)
ATT={'Square':E.square_attack,'SimBA':E.simba_dct_attack,'SignH':E.signhunter_attack}
s1={}; sw=0; sn=0
for an,af in ATT.items():
    succ=0
    for k,idx in enumerate(idxs):
        srng=np.random.RandomState(3000+k); o=Oracle(0.5,srng); o.reset(0)
        np.random.seed(4000+k); x,y=testset[int(idx)]; x=x.numpy()
        _,_,s=af(o,x,int(y),1000); succ+=int(s); sw+=o.switches; sn+=1
    s1[an]=succ/200*100
s1['Avg']=np.mean([s1[a] for a in ATT]); s1['Sw']=sw/sn
out['setting1']=s1
print(f"[S1] Square{s1['Square']:.1f} SimBA{s1['SimBA']:.1f} SignH{s1['SignH']:.1f} Avg{s1['Avg']:.1f} Sw{s1['Sw']:.1f}",flush=True)
json.dump(out,open('transformer_paper.json','w'),indent=2)

# ---- Setting 2: run_matched replica (benign 2000, targets 300) ----
glob=np.random.RandomState(7); targets=glob.choice(correct,300,replace=False)
benign2=[testset[int(i)][0].numpy() for i in glob.randint(0,len(testset),size=2000)]; BURST=250; STREAM=1000
def benign_fsw(thr):
    tot=0
    for b in range(40):
        srng=np.random.RandomState(7000+b); o=Oracle(thr,srng); o.reset(0)
        for _ in range(1000): o(benign2[srng.randint(0,len(benign2))][None])
        tot+=o.switches
    return tot/40/1000*1000
def burst_asr(thr):
    succ=0
    for b in range(120):
        srng=np.random.RandomState(5000+b); o=Oracle(thr,srng); o.reset(0)
        cp=srng.randint(0,STREAM-BURST)
        for _ in range(cp): o(benign2[srng.randint(0,len(benign2))][None])
        np.random.seed(9000+b); idx=int(srng.choice(targets)); x,y=testset[idx]; x=x.numpy()
        _,_,s=E.signhunter_attack(o,x,int(y),o.query_count+min(BURST,STREAM-o.query_count)); succ+=int(s)
    return succ/120*100
for name,thr in [('perf',0.5),('evasion',0.15)]:
    out['setting2_'+name]=[benign_fsw(thr),burst_asr(thr)]
    print(f"[S2 {name}] fsw={out['setting2_'+name][0]:.2f} burst={out['setting2_'+name][1]:.1f}",flush=True)
    json.dump(out,open('transformer_paper.json','w'),indent=2)

# ---- Setting 3: run_tableB replica (benign 4000, 24 streams) ----
glob=np.random.RandomState(7); benign4=[testset[int(i)][0].numpy() for i in glob.randint(0,len(testset),size=4000)]
MIL=[1000,4000,16000]; cur=np.zeros(3)
for b in range(24):
    srng=np.random.RandomState(7000+b); o=Oracle(0.5,srng); o.reset(0); mi=0
    for q in range(1,16001):
        o(benign4[srng.randint(0,len(benign4))][None])
        if mi<3 and q==MIL[mi]: cur[mi]+=o.switches; mi+=1
out['setting3']=(cur/24).tolist()
print(f"[S3] accum@{MIL} = {out['setting3']}",flush=True)
json.dump(out,open('transformer_paper.json','w'),indent=2)
print('[done]',flush=True)
