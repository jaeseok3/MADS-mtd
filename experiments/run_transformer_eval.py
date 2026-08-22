"""End-to-end evaluation of the Transformer confirmer vs LSTM/FCN in all three
settings (same seeded harness as the paper tables), to test whether the
Transformer actually detects attacks better and/or cuts false switches.
All three confirmers use gate avg-10<0.6, threshold p_thr=0.5, greedy selection."""
import sys, os, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque
from torchvision import datasets, transforms
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_switching as E
CKPT=os.path.dirname(os.path.abspath(__file__)); W=10; L=50; QMAX=1000; NIMG=200; BURST=250; STREAM=1000
device=torch.device('cuda')
pool=E.load_pool('divergent_kd_v5b_pool',device); K=len(pool)

class FCN(nn.Module):
    def __init__(s): super().__init__(); s.n=nn.Sequential(nn.Linear(6,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(s,x): return s.n(x).squeeze(1)
class LSTMc(nn.Module):
    def __init__(s): super().__init__(); s.lstm=nn.LSTM(1,32,batch_first=True); s.fc=nn.Linear(32,1)
    def forward(s,x): o,_=s.lstm(x.unsqueeze(-1)); return s.fc(o[:,-1]).squeeze(1)
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
fcn=FCN(); fcn.load_state_dict(torch.load(os.path.join(CKPT,'dl_fcn.pth'),map_location='cpu')); fcn.eval()
lstm=LSTMc(); lstm.load_state_dict(torch.load(os.path.join(CKPT,'dl_lstm.pth'),map_location='cpu')); lstm.eval()
trf=Tr(); trf.load_state_dict(torch.load(os.path.join(CKPT,'dl_transformer.pth'),map_location='cpu')); trf.eval()
def win_feats(w):
    return torch.tensor(np.array([w.mean(),w.min(),w[-1],(w<0.3).mean(),np.polyfit(np.arange(L),w,1)[0]*L,w.std()],dtype=np.float32)).unsqueeze(0)
def confirm(kind,w):
    t=torch.tensor(w,dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        if kind=='fcn': return torch.sigmoid(fcn(win_feats(w))).item()
        if kind=='lstm': return torch.sigmoid(lstm(t)).item()
        return torch.sigmoid(trf(t)).item()

class Oracle:
    def __init__(s,kind,thr,rng): s.kind=kind; s.thr=thr; s.rng=rng
    def reset(s,start=0): s.cur=start; s.query_count=0; s.switches=0; s.hist=deque(maxlen=L)
    def __call__(s,x):
        if isinstance(x,np.ndarray): x=torch.from_numpy(x).float()
        xt=x.to(device)
        with torch.no_grad(): sc=F.softmax(pool[s.cur](xt),dim=1).cpu().numpy()[0]
        ss=np.sort(sc)[::-1]; s.hist.append(ss[0]-ss[1]); s.query_count+=1; h=list(s.hist); fire=False
        if len(h)>=W and np.mean(h[-W:])<0.6:
            w=np.array(h[-L:])
            if len(w)<L: w=np.pad(w,(L-len(w),0),constant_values=w[0])
            fire=confirm(s.kind,w)>s.thr
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
correct=np.array(correct); glob=np.random.RandomState(7)
idxs=glob.choice(correct,NIMG,replace=False)           # Setting 1
targets=glob.choice(correct,300,replace=False)          # Setting 2 bursts
benign=[testset[int(i)][0].numpy() for i in glob.randint(0,len(testset),size=4000)]
ATTACKS={'Square':E.square_attack,'SimBA':E.simba_dct_attack,'SignH':E.signhunter_attack}
THR=0.5

def setting1(kind):
    out={}; sw=0; sn=0
    for an,af in ATTACKS.items():
        succ=0
        for k,idx in enumerate(idxs):
            srng=np.random.RandomState(3000+k); o=Oracle(kind,THR,srng); o.reset(0)
            np.random.seed(4000+k); x,y=testset[int(idx)]; x=x.numpy()
            _,_,s=af(o,x,int(y),QMAX); succ+=int(s); sw+=o.switches; sn+=1
        out[an]=succ/NIMG*100
    out['Avg']=np.mean([out[a] for a in ATTACKS]); out['Sw']=sw/sn; return out

def setting2(kind):
    # benign false-switch /1000q
    tot=0
    for b in range(40):
        srng=np.random.RandomState(7000+b); o=Oracle(kind,THR,srng); o.reset(0)
        for _ in range(1000): o(benign[srng.randint(0,len(benign))][None])
        tot+=o.switches
    fsw=tot/40/1000*1000
    # burst ASR
    succ=0
    for b in range(120):
        srng=np.random.RandomState(5000+b); o=Oracle(kind,THR,srng); o.reset(0)
        cp=srng.randint(0,STREAM-BURST)
        for _ in range(cp): o(benign[srng.randint(0,len(benign))][None])
        np.random.seed(9000+b); idx=int(srng.choice(targets)); x,y=testset[idx]; x=x.numpy()
        _,_,s=E.signhunter_attack(o,x,int(y),o.query_count+min(BURST,STREAM-o.query_count)); succ+=int(s)
    return fsw, succ/120*100

def setting3(kind):
    MIL=[1000,4000,16000]; cur=np.zeros(3)
    for b in range(12):
        srng=np.random.RandomState(7000+b); o=Oracle(kind,THR,srng); o.reset(0); mi=0
        for q in range(1,16001):
            o(benign[srng.randint(0,len(benign))][None])
            if mi<3 and q==MIL[mi]: cur[mi]+=o.switches; mi+=1
    return (cur/12).tolist()

res={}
for kind in ['fcn','lstm','transformer']:
    s1=setting1(kind); fsw,basr=setting2(kind); acc=setting3(kind)
    res[kind]={'pure':s1,'mixed_fsw':fsw,'mixed_burst':basr,'benign_accum':acc}
    print(f"[{kind:11s}] pure Avg={s1['Avg']:.1f} (Sq{s1['Square']:.0f}/Si{s1['SimBA']:.0f}/SH{s1['SignH']:.0f}) Sw={s1['Sw']:.1f} | mixed fsw={fsw:.2f} burst={basr:.1f} | benign@16k={acc[2]:.2f}",flush=True)
    json.dump(res,open('transformer_eval.json','w'),indent=2)
print('[done]',flush=True)
