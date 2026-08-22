"""Generalized Table-7 harness: adaptive (switch-aware re-baselining SignHunter,
Q=2000) score-based ASR of every defense (None/AAA/Stateful/Blacklight/MADS+shield)
for ONE (dataset, arch). Same settings as the CIFAR-10 ResNet anchor
(c10_adaptive_defenses.py): NIMG=150, shield alpha_max=16 tau_sh=800.
Decision-based (HSJA) ASR comes separately from arch_hardlabel_{ds}_{arch}.json.
-> adaptivedef_{DS}_{ARCH}.json
Usage: python3 adaptive_defenses_gen.py <ds:c10|cifar|tin> <arch>
"""
import os, sys, json
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_common as A

DS, ARCH = sys.argv[1], sys.argv[2]
device = torch.device('cuda')
W, L = 10, 50
GATE = 0.85 if DS == 'c10' else 0.6
EPS = 8/255
NIMG = 150
MAXQ = 2000
SHIELD_AMAX = 16.0
SHIELD_TAU = 800.0

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


def shield(z, tsw, amax, tau, mref=4.0):
    c = int(z.argmax()); zs = np.sort(z)[::-1]; margin = zs[0]-zs[1]
    alpha = 0.03 + (amax-0.03)*np.exp(-tsw/tau)
    strength = alpha*np.clip(margin/mref, 0.0, 1.0)
    noise = np.random.RandomState(((tsw*7+c) % (2**31))).normal(0, strength, size=z.shape).astype(np.float32)
    noise[c] = 0.0; zp = z + noise
    if c != int(zp.argmax()): zp[c] = zp.max()+1e-3
    return zp


feat_hold = {}
def _hook(m, i, o): feat_hold['f'] = i[0].detach()
lin = None
for mod in POOL[0].modules():
    if isinstance(mod, nn.Linear): lin = mod
lin.register_forward_hook(_hook)
def embed(xt):
    with torch.no_grad(): POOL[0](xt)
    return feat_hold['f'][0].cpu().numpy()


def fingerprint(x, Lq=16, P=8):
    C, H, Wd = x.shape
    xq = np.clip((x*Lq).astype(np.int16), 0, Lq-1); hs = set()
    for c in range(C):
        for i in range(0, H, P):
            for j in range(0, Wd, P):
                hs.add(hash((c, i, j, xq[c, i:i+P, j:j+P].tobytes())))
    return hs


def fwd_scores(cur, xt):
    with torch.no_grad(): return POOL[cur](xt).cpu().numpy()[0]


class Defense:
    def __init__(s, kind, thr=None, amax=0.0, tau=SHIELD_TAU):
        s.kind = kind; s.thr = thr; s.amax = amax; s.tau = tau
    def reset(s, start=0):
        s.cur = start; s.query_count = 0; s.switches = 0
        s.hist = deque(maxlen=L); s.tsw = 10**8
        s.buf = deque(maxlen=500); s.bl_idx = {}; s.bl_q = 0
    def __call__(s, x):
        xnp = x[0] if (isinstance(x, np.ndarray) and x.ndim == 4) else x
        xt = (torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x)
        if xt.dim() == 3: xt = xt[None]
        xt = xt.to(device); s.query_count += 1; s.tsw += 1
        z = fwd_scores(s.cur, xt)
        if s.kind == 'aaa':
            c = int(z.argmax()); zo = z.copy(); zo[c] = -1e9; m2 = zo.max(); l = z[c]-m2
            lp = max(l - 2.0*np.sin(2*np.pi*l/4.0), 0.05); z2 = z.copy(); z2[c] = m2+lp
            return F.softmax(torch.tensor(z2), 0).numpy()
        if s.kind == 'none':
            return F.softmax(torch.tensor(z), 0).numpy()
        sc_true = F.softmax(torch.tensor(z), 0).numpy(); ss = np.sort(sc_true)[::-1]
        s.hist.append(ss[0]-ss[1]); h = list(s.hist); fire = False
        if s.kind == 'mads_shield':
            if len(h) >= W and np.mean(h[-W:]) < GATE:
                w = np.array(h[-L:]); w = np.pad(w, (L-len(w), 0), constant_values=w[0]) if len(w) < L else w
                fire = conf_tr(w) > 0.5
        elif s.kind == 'stateful':
            e = embed(xt)
            if len(s.buf) >= 10:
                B = np.stack(s.buf); d = np.linalg.norm(B-e, axis=1); kk = min(5, len(d)); dk = np.sort(d)[:kk].mean()
                if dk < s.thr: fire = True
            s.buf.append(e)
        elif s.kind == 'blacklight':
            fp = fingerprint(xnp if isinstance(xnp, np.ndarray) else xnp.cpu().numpy()); s.bl_q += 1
            cnt = Counter()
            for hh in fp:
                if hh in s.bl_idx: cnt.update(s.bl_idx[hh])
            if cnt and max(cnt.values())/len(fp) >= s.thr: fire = True
            for hh in fp: s.bl_idx.setdefault(hh, []).append(s.bl_q)
        if fire:
            s.cur = greedy_pick(xt, s.cur); s.switches += 1; s.hist.clear(); s.tsw = 0
        if s.kind == 'mads_shield' and s.amax > 0:
            z = shield(z, s.tsw, s.amax, s.tau)
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
                if oracle.query_count-last >= 30:
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


def calib_thr(kind, cands, bimgs):
    best = None
    for thr in cands:
        fsw = 0
        for t in range(6):
            o = Defense(kind, thr=thr); o.reset(0)
            for k in range(min(1000, len(bimgs))): o(bimgs[k % len(bimgs)])
            fsw += o.switches
        f = fsw/6
        if best is None: best = (thr, f)
        if f <= 0.5: best = (thr, f); break
    return best


def main():
    rng = np.random.RandomState(7)
    correct, labs = A.correct_by_all(POOL, ts, device)
    idxs = rng.choice(correct, NIMG, replace=False)
    imgs = [A.img_np(ts, i)[0] for i in idxs]; ys = [int(labs[i]) for i in idxs]
    bidx = rng.choice(len(ts), 1200, replace=False)
    bimgs = [A.img_np(ts, i)[0] for i in bidx]

    st_thr, st_f = calib_thr('stateful', [2.0, 1.5, 1.0, 0.7, 0.5], bimgs)
    bl_thr, bl_f = calib_thr('blacklight', [0.5, 0.35, 0.25, 0.15], bimgs)
    print(f"  [{DS}/{ARCH} calib] stateful thr={st_thr} fsw={st_f:.2f} | blacklight thr={bl_thr} fsw={bl_f:.2f}", flush=True)

    DEFS = [
        ('None (static)', lambda: Defense('none')),
        ('AAA',           lambda: Defense('aaa')),
        ('Stateful Det.', lambda: Defense('stateful', thr=st_thr)),
        ('Blacklight',    lambda: Defense('blacklight', thr=bl_thr)),
        ('MADS',          lambda: Defense('mads_shield', amax=SHIELD_AMAX)),
    ]
    res = {}
    for name, mk in DEFS:
        succ = 0
        for k in range(NIMG):
            o = mk(); o.reset(0)
            succ += int(adaptive_signhunter(o, imgs[k], ys[k], MAXQ))
        asr = round(succ/NIMG*100, 1)
        res[name] = asr
        print(f"  [{DS}/{ARCH}] {name:14s} score_asr={asr:5.1f}", flush=True)
        json.dump(res, open(f'adaptivedef_{DS}_{ARCH}.json', 'w'), indent=2)
    print(f"[done] {DS}/{ARCH} {res}", flush=True)


if __name__ == '__main__':
    main()
