"""Try a Transformer confirmer (and a classical GBM) vs FCN/LSTM on the same
labeled margin-window task. Reports held-out (val) accuracy/AUC with an 80/20
split, so we can decide whether to switch the confirmer to Transformer/ML.
Saves dl_transformer.pth if it wins."""
import sys, os, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque
from torchvision import datasets, transforms
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_switching as E
POOL_DIR='divergent_kd_v5b_pool'; TAU_GATE=0.6; W=10; L=50
CKPT=os.path.dirname(os.path.abspath(__file__))
device=torch.device('cuda')
pool=E.load_pool(POOL_DIR,device); K=len(pool)
tf=transforms.Compose([transforms.ToTensor()])
testset=datasets.CIFAR100(root=E.DATA_ROOT,train=False,download=False,transform=tf)
loader=torch.utils.data.DataLoader(testset,batch_size=256,shuffle=False,num_workers=2)
correct,off=[],0
with torch.no_grad():
    for im,la in loader:
        im,la=im.to(device),la.to(device); bc=torch.ones(la.size(0),dtype=torch.bool,device=device)
        for m in pool: bc&=(m(im).argmax(1)==la)
        for j in torch.nonzero(bc).squeeze(1).tolist(): correct.append(off+j)
        off+=la.size(0)
correct=np.array(correct)

def windows_from_stream(margins):
    out=[]
    for t in range(W-1,len(margins)):
        if t<L-1:
            w=np.array(margins[:t+1]); w=np.pad(w,(L-len(w),0),constant_values=w[0] if len(w) else 1.0)
        else: w=np.array(margins[t-L+1:t+1])
        if np.mean(w[-W:])<TAU_GATE: out.append(w.astype(np.float32))
    return out

rng=np.random.RandomState(0); benign_w=[]
class Rec:
    def __init__(s): s.cur=0
    def __call__(s,x):
        if isinstance(x,np.ndarray): x=torch.from_numpy(x).float()
        with torch.no_grad(): sc=F.softmax(pool[0](x.to(device)),dim=1).cpu().numpy()
        return sc
for _ in range(120):
    ms=[]; o=Rec()
    for i in rng.randint(0,len(testset),size=300):
        sc=o(testset[int(i)][0].numpy()[None]); ss=np.sort(sc[0])[::-1]; ms.append(ss[0]-ss[1])
    benign_w+=windows_from_stream(ms)
atk_w=[]; sel=rng.choice(correct,200,replace=False)
for ix in sel:
    x,y=testset[int(ix)]; x=x.numpy(); ms=[]
    class Rec2:
        def __init__(s2): s2.query_count=0; s2.cur=0
        def reset(s2,start_idx=0): s2.query_count=0
        def __call__(s2,xx):
            if isinstance(xx,np.ndarray): xx=torch.from_numpy(xx).float()
            with torch.no_grad(): sc=F.softmax(pool[0](xx.to(device)),dim=1).cpu().numpy()
            s2.query_count+=xx.shape[0]; ss=np.sort(sc[0])[::-1]; ms.append(ss[0]-ss[1]); return sc
    E.signhunter_attack(Rec2(),x,int(y),300); atk_w+=windows_from_stream(ms)
print(f"[*] benign={len(benign_w)} attack={len(atk_w)}",flush=True)

X=np.concatenate([np.array(benign_w),np.array(atk_w)])
Y=np.concatenate([np.zeros(len(benign_w)),np.ones(len(atk_w))]).astype(np.float32)
perm=rng.permutation(len(X)); X=X[perm]; Y=Y[perm]
nval=len(X)//5; Xtr,Xva=X[nval:],X[:nval]; Ytr,Yva=Y[nval:],Y[:nval]
def feats(m): return np.stack([m.mean(1),m.min(1),m[:,-1],((m<0.3).mean(1)),
    np.array([np.polyfit(np.arange(L),r,1)[0]*L for r in m]),m.std(1)],1).astype(np.float32)

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

def run(model,Xtr_in,Xva_in,name,epochs=40):
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); n=len(Xtr_in); idx=np.arange(n)
    yt=torch.tensor(Ytr); yv=torch.tensor(Yva)
    for ep in range(epochs):
        model.train(); np.random.shuffle(idx)
        for b in range(0,n,256):
            bi=idx[b:b+256]; logit=model(Xtr_in[bi]); loss=F.binary_cross_entropy_with_logits(logit,yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pv=torch.sigmoid(model(Xva_in)).numpy()
    acc=((pv>0.5).astype(float)==Yva).mean()
    # AUC
    order=np.argsort(pv); r=np.empty_like(order,dtype=float); r[order]=np.arange(len(pv))
    npos=Yva.sum(); nneg=len(Yva)-npos
    auc=(r[Yva==1].sum()-npos*(npos-1)/2)/(npos*nneg) if npos*nneg>0 else float('nan')
    print(f"  [{name:11s}] val acc={acc*100:.1f}%  val AUC={auc:.3f}",flush=True)
    return model,acc,auc

Xtr_f=torch.tensor(feats(Xtr)); Xva_f=torch.tensor(feats(Xva))
Xtr_t=torch.tensor(Xtr); Xva_t=torch.tensor(Xva)
res={}
_,a,u=run(FCN(),Xtr_f,Xva_f,'FCN'); res['FCN']=[a,u]
_,a,u=run(LSTMc(),Xtr_t,Xva_t,'LSTM'); res['LSTM']=[a,u]
trm,a,u=run(Tr(),Xtr_t,Xva_t,'Transformer'); res['Transformer']=[a,u]
torch.save(trm.state_dict(),os.path.join(CKPT,'dl_transformer.pth'))
# classical ML
try:
    from sklearn.ensemble import GradientBoostingClassifier
    gb=GradientBoostingClassifier(n_estimators=200,max_depth=3).fit(feats(Xtr),Ytr)
    pv=gb.predict_proba(feats(Xva))[:,1]; acc=((pv>0.5).astype(float)==Yva).mean()
    from sklearn.metrics import roc_auc_score; auc=roc_auc_score(Yva,pv)
    print(f"  [{'GBM(ML)':11s}] val acc={acc*100:.1f}%  val AUC={auc:.3f}",flush=True); res['GBM']=[acc,auc]
except Exception as e:
    print('  [GBM] skipped:',e,flush=True)
json.dump(res,open('confirmer_compare.json','w'),indent=2)
print('[done]',flush=True)
