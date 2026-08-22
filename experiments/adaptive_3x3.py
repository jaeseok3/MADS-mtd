"""3x3 (dataset x architecture) adaptive-robustness of MADS+obfuscator, on ONE
consistent hardened harness (switch-aware re-baselining SignHunter, Q=2000,
alpha_max=16, tau_sh=800, NIMG=150) -- identical settings to the CIFAR-10 ResNet
Table-7 anchor (24.7). Reports MADS+obfuscator adaptive score-based ASR (%) and benign
clean acc. Decision-based (HSJA) MADS ASR is already 0 across all 9 combos
(arch_hardlabel_{ds}_{arch}.json, HSJA_ours). -> adaptive3x3_{DS}_{ARCH}.json

Usage: python3 adaptive_3x3.py <ds:c10|cifar|tin> <arch:resnet|vgg|alexnet>
"""
import os, sys, json
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_common as A

DS, ARCH = sys.argv[1], sys.argv[2]
device = torch.device('cuda')
W, L = 10, 50
GATE = 0.85 if DS == 'c10' else 0.6
EPS = 8/255
NIMG = 150
MAXQ = 2000
OBF_AMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 16.0
OBF_TAU = float(sys.argv[4]) if len(sys.argv) > 4 else 800.0

POOLTAGS = {
    'c10':   {'resnet': 'c10_div_kd_resnet', 'vgg': 'c10_div_kd_vgg_final', 'alexnet': 'c10_div_kd_alexnet_final'},
    'cifar': {'resnet': 'divergent_kd_v5b_pool', 'vgg': 'divkd_b4.0_t10.0_vgg16', 'alexnet': 'divkd_b4.0_t10.0_alexnet'},
    'tin':   {'resnet': 'tin_div_kd', 'vgg': 'tin_div_kd_vgg', 'alexnet': 'tin_div_kd_alexnet'},
}
POOL = A.load_pool(POOLTAGS[DS][ARCH], ARCH, DS, device); K = len(POOL)
ts = A.get_testset(DS)
CJT = os.path.join(A.CHECKPOINT_DIR, f'{DS}_{ARCH}')


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


def obfuscator(z, tsw, amax, tau, mref=4.0):
    c = int(z.argmax()); zs = np.sort(z)[::-1]; margin = zs[0]-zs[1]
    alpha = 0.03 + (amax-0.03)*np.exp(-tsw/tau)
    strength = alpha*np.clip(margin/mref, 0.0, 1.0)
    noise = np.random.RandomState(((tsw*7+c) % (2**31))).normal(0, strength, size=z.shape).astype(np.float32)
    noise[c] = 0.0; zp = z + noise
    if c != int(zp.argmax()): zp[c] = zp.max()+1e-3
    return zp


def fwd_scores(cur, xt):
    with torch.no_grad(): return POOL[cur](xt).cpu().numpy()[0]


class MADS:
    def __init__(s, amax=OBF_AMAX, tau=OBF_TAU):
        s.amax = amax; s.tau = tau
    def reset(s, start=0):
        s.cur = start; s.query_count = 0; s.switches = 0
        s.hist = deque(maxlen=L); s.tsw = 10**8
    def __call__(s, x):
        xt = (torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x)
        if xt.dim() == 3: xt = xt[None]
        xt = xt.to(device); s.query_count += 1; s.tsw += 1
        z = fwd_scores(s.cur, xt)
        sc_true = F.softmax(torch.tensor(z), 0).numpy(); ss = np.sort(sc_true)[::-1]
        s.hist.append(ss[0]-ss[1]); h = list(s.hist); fire = False
        if len(h) >= W and np.mean(h[-W:]) < GATE:
            w = np.array(h[-L:]); w = np.pad(w, (L-len(w), 0), constant_values=w[0]) if len(w) < L else w
            fire = conf_tr(w) > 0.5
        if fire:
            s.cur = greedy_pick(xt, s.cur); s.switches += 1; s.hist.clear(); s.tsw = 0
        if s.amax > 0:
            z = obfuscator(z, s.tsw, s.amax, s.tau)
        return F.softmax(torch.tensor(z), 0).numpy()


def clip_eps(xa, xo, eps): return np.clip(np.clip(xa, xo-eps, xo+eps), 0, 1)


def adaptive_signhunter(oracle, x0, y, maxq):
    C, H, Wd = x0.shape; dim = C*H*Wd; sign = np.ones(dim, np.float32)
    xadv = clip_eps(x0+EPS*sign.reshape(C, H, Wd), x0, EPS)
    sc = oracle(xadv)
    if np.argmax(sc) != y: return True
    best = sc[y]; rng = np.random.RandomState(0); last = 0
    while oracle.query_count < maxq:
        perm = rng.permutation(dim); chunk = dim
        while oracle.query_count < maxq and chunk >= 1:
            for st in range(0, dim, chunk):
                if oracle.query_count >= maxq: break
                if oracle.query_count-last >= 30:      # switch-aware re-baseline
                    sc = oracle(xadv); last = oracle.query_count
                    if np.argmax(sc) != y: return True
                    best = sc[y]
                    if oracle.query_count >= maxq: break
                idx = perm[st:st+chunk]; sn = sign.copy(); sn[idx] *= -1
                xn = clip_eps(x0+EPS*sn.reshape(C, H, Wd), x0, EPS)
                sc = oracle(xn)
                if np.argmax(sc) != y: return True
                if sc[y] < best: sign = sn; best = sc[y]; xadv = xn
            chunk //= 2
    return np.argmax(oracle(xadv)) != y


def main():
    rng = np.random.RandomState(7)
    correct, labs = A.correct_by_all(POOL, ts, device)
    idxs = rng.choice(correct, NIMG, replace=False)
    imgs = [A.img_np(ts, i)[0] for i in idxs]; ys = [int(labs[i]) for i in idxs]
    bidx = rng.choice(len(ts), 200, replace=False)
    bimgs = [A.img_np(ts, i)[0] for i in bidx]; bys = [A.img_np(ts, i)[1] for i in bidx]

    succ = 0
    for k in range(NIMG):
        o = MADS(); o.reset(0)
        succ += int(adaptive_signhunter(o, imgs[k], ys[k], MAXQ))
        if (k+1) % 30 == 0: print(f"  [{DS}/{ARCH}] {k+1}/{NIMG} running_ASR={succ/(k+1)*100:.1f}", flush=True)
    asr = succ/NIMG*100

    corr = 0
    for k in range(50):
        o = MADS(); o.reset(0); sc = o(bimgs[k]); corr += int(np.argmax(sc) == bys[k])
    bacc = corr/50*100

    out = {'ds': DS, 'arch': ARCH, 'score_asr': round(asr, 1), 'dec_asr': 0.0,
           'benign_acc': round(bacc, 1), 'nimg': NIMG, 'amax': OBF_AMAX, 'tau': OBF_TAU}
    _tag = '' if (OBF_AMAX == 16.0 and OBF_TAU == 800.0) else f'_a{int(OBF_AMAX)}t{int(OBF_TAU)}'
    json.dump(out, open(f'adaptive3x3_{DS}_{ARCH}{_tag}.json', 'w'), indent=2)
    print(f"[done] {DS}/{ARCH} score_asr={asr:.1f} dec_asr=0.0 benign_acc={bacc:.1f}", flush=True)


if __name__ == '__main__':
    main()
