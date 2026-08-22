"""Adaptive-robustness campaign for the switch-triggered decaying shield, per
architecture (CIFAR-10). For a sweep of shield strength alpha (tau fixed), under
MADS switching, measures:
  restart_asr  : switch-aware SignHunter that RE-BASELINES (re-attacks each new
                 model) -- the strategy the shield is meant to defeat.
  transfer_asr : SignHunter that does NOT re-baseline -- it reuses its accumulated
                 perturbation across switches (transfer-fallback) -- defeated by
                 pool divergence, not the shield.
  acc          : benign returned-label accuracy (top-1 preserved => shield is
                 free). Also benign false switches.
Usage: python c10_adaptive_campaign.py --arch resnet
-> c10_adaptive_campaign_<arch>.json
"""
import os, sys, json, argparse
import numpy as np, torch
import torch.nn.functional as F
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_common as A
import torch.nn as nn, pickle

device = torch.device('cuda')
W, L, GATE = 10, 50, 0.85
EPS = 8/255
ALPHAS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
TAU = 100.0
NIMG = 60
MAXQ = 2000

ap = argparse.ArgumentParser(); ap.add_argument('--arch', required=True); args = ap.parse_args()
ARCH = args.arch
POOLTAG = {'resnet': 'c10_div_kd_resnet', 'vgg': 'c10_div_kd_vgg_final', 'alexnet': 'c10_div_kd_alexnet_final'}[ARCH]
POOL = A.load_pool(POOLTAG, ARCH, 'c10', device); K = len(POOL)
ts = A.get_testset('c10')
CJT = os.path.join(A.CHECKPOINT_DIR, f'c10_{ARCH}')
gbm = pickle.load(open(CJT+'_gbm.pkl', 'rb'))

class Tr(nn.Module):
    def __init__(s, d=32, nhead=4, nl=2):
        super().__init__(); s.emb = nn.Linear(1, d)
        pe = torch.zeros(L, d); pos = torch.arange(0, L).float().unsqueeze(1)
        dt = torch.exp(torch.arange(0, d, 2).float()*(-np.log(10000.0)/d))
        pe[:, 0::2] = torch.sin(pos*dt); pe[:, 1::2] = torch.cos(pos*dt); s.register_buffer('pe', pe)
        el = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=64, batch_first=True, dropout=0.0)
        s.enc = nn.TransformerEncoder(el, nl); s.fc = nn.Linear(d, 1)
    def forward(s, w): h = s.emb(w[:, :, None])+s.pe[None]; return s.fc(s.enc(h).mean(1))[:, 0]
trf = Tr().to(device).eval(); trf.load_state_dict(torch.load(CJT+'_transformer.pth', map_location=device)); trf.eval()

def conf_tr(w):
    with torch.no_grad(): return torch.sigmoid(trf(torch.tensor(w, dtype=torch.float32, device=device)[None])).item()
def greedy_pick(xt, cur):
    ms = []
    with torch.no_grad():
        for m in POOL: c = F.softmax(m(xt), 1).cpu().numpy()[0]; cs = np.sort(c)[::-1]; ms.append(cs[0]-cs[1])
    others = [i for i in range(K) if i != cur]; return max(others, key=lambda i: ms[i])

def shield(z, tsw, amax, tau, mref=4.0):
    c = int(z.argmax()); zs = np.sort(z)[::-1]; margin = zs[0]-zs[1]
    alpha = 0.03 + (amax-0.03)*np.exp(-tsw/tau)
    strength = alpha*np.clip(margin/mref, 0.0, 1.0)
    noise = np.random.RandomState(((tsw*7+c) % (2**31))).normal(0, strength, size=z.shape).astype(np.float32)
    noise[c] = 0.0; zp = z + noise
    if c != int(zp.argmax()): zp[c] = zp.max()+1e-3
    return zp

class Oracle:
    def __init__(s, rng, amax=0.0, tau=TAU): s.rng = rng; s.amax = amax; s.tau = tau
    def reset(s, start=0):
        s.cur = start; s.query_count = 0; s.switches = 0; s.hist = deque(maxlen=L); s.tsw = 10**8
    def __call__(s, x):
        xt = (torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x).to(device)
        with torch.no_grad(): z = POOL[s.cur](xt).cpu().numpy()[0]
        sc_true = F.softmax(torch.tensor(z), 0).numpy(); ss = np.sort(sc_true)[::-1]
        s.hist.append(ss[0]-ss[1]); s.query_count += 1; s.tsw += 1
        h = list(s.hist); fire = False
        if len(h) >= W and np.mean(h[-W:]) < GATE:
            w = np.array(h[-L:]); w = np.pad(w, (L-len(w), 0), constant_values=w[0]) if len(w) < L else w
            fire = conf_tr(w) > 0.5
        if fire:
            s.cur = greedy_pick(xt, s.cur); s.switches += 1; s.hist.clear(); s.tsw = 0
        if s.amax > 0: z = shield(z, s.tsw, s.amax, s.tau)
        return F.softmax(torch.tensor(z), 0).numpy()

def clip_eps(xa, xo, eps): return np.clip(np.clip(xa, xo-eps, xo+eps), 0, 1)

def signhunter(oracle, x0, y, maxq, rebaseline):
    C, H, Wd = x0.shape; dim = C*H*Wd; sign = np.ones(dim, np.float32)
    xadv = clip_eps(x0+EPS*sign.reshape(C, H, Wd), x0, EPS)
    sc = oracle(xadv[None] if xadv.ndim == 3 else xadv)
    if np.argmax(sc) != y: return True
    best = sc[y]; rng = np.random.RandomState(0); last = 0
    while oracle.query_count < maxq:
        perm = rng.permutation(dim); chunk = dim
        while oracle.query_count < maxq and chunk >= 1:
            for st in range(0, dim, chunk):
                if oracle.query_count >= maxq: break
                if rebaseline and oracle.query_count-last >= 30:
                    sc = oracle(xadv[None] if xadv.ndim == 3 else xadv); last = oracle.query_count
                    if np.argmax(sc) != y: return True
                    best = sc[y]
                    if oracle.query_count >= maxq: break
                idx = perm[st:st+chunk]; sn = sign.copy(); sn[idx] *= -1
                xn = clip_eps(x0+EPS*sn.reshape(C, H, Wd), x0, EPS)
                sc = oracle(xn[None] if xn.ndim == 3 else xn)
                if np.argmax(sc) != y: return True
                if sc[y] < best: sign = sn; best = sc[y]; xadv = xn
            chunk //= 2
    return np.argmax(oracle(xadv[None] if xadv.ndim == 3 else xadv)) != y

def benign_acc(amax, bimgs, bys):
    corr = 0; fsw = 0; trials = 15
    for t in range(trials):
        o = Oracle(np.random.RandomState(t), amax); o.reset(0)
        for k in range(len(bimgs)):
            sc = o(bimgs[k]); corr += int(np.argmax(sc) == bys[k])
        fsw += o.switches
    return corr/(trials*len(bimgs))*100, fsw/trials

def main():
    rng = np.random.RandomState(7)
    correct, labs = A.correct_by_all(POOL, ts, device)
    idxs = rng.choice(correct, NIMG, replace=False)
    imgs = [A.img_np(ts, i)[0] for i in idxs]; ys = [int(labs[i]) for i in idxs]
    bidx = rng.choice(len(ts), 50, replace=False)
    bimgs = [A.img_np(ts, i)[0] for i in bidx]; bys = [A.img_np(ts, i)[1] for i in bidx]
    res = {}
    for amax in ALPHAS:
        r_succ = t_succ = 0
        for k in range(NIMG):
            o = Oracle(np.random.RandomState(1000+k), amax); o.reset(0)
            r_succ += int(signhunter(o, imgs[k], ys[k], MAXQ, rebaseline=True))
            o2 = Oracle(np.random.RandomState(1000+k), amax); o2.reset(0)
            t_succ += int(signhunter(o2, imgs[k], ys[k], MAXQ, rebaseline=False))
        acc, fsw = benign_acc(amax, bimgs, bys)
        res[f'{amax}'] = {'restart_asr': r_succ/NIMG*100, 'transfer_asr': t_succ/NIMG*100,
                          'acc': round(acc, 1), 'benign_fsw': round(fsw, 2)}
        print(f"  {ARCH} a={amax:.0f}  restart={r_succ/NIMG*100:.1f}  transfer={t_succ/NIMG*100:.1f}  acc={acc:.1f}  fsw={fsw:.2f}", flush=True)
        json.dump(res, open(f'c10_adaptive_campaign_{ARCH}.json', 'w'), indent=2)
    print("[done]", flush=True)

if __name__ == '__main__':
    main()
