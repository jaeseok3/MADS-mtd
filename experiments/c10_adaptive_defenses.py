"""Table 7 harness: adaptive-attack ASR of NON-MADS defenses vs MADS, on ONE
consistent harness (CIFAR-10 ResNet, switch-aware re-baselining SignHunter,
Q=2000, same target images). The adaptive attacker (Algorithm 2) re-baselines its
reference to the LIVE deployed output every 30 queries -- the strategy that
defeats plain switching. We run it against:
  None (static single model), RND (input noise), AAA (logit shaping, top-1 kept),
  Stateful Detection / Blacklight (adapted as switch triggers + greedy select),
  MADS (switching only), MADS+obfuscator (switch-triggered decaying output obfuscator).
Reports adaptive ASR (%) and benign clean accuracy (%). -> c10_adaptive_defenses.json
"""
import os, sys, json, pickle
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_common as A

device = torch.device('cuda')
W, L, GATE = 10, 50, 0.85
EPS = 8/255
NIMG = 150
MAXQ = 2000
OBF_AMAX = 16.0     # accuracy is flat for any alpha (campaign); use strong point
OBF_TAU = 800.0

POOL = A.load_pool('c10_div_kd_resnet', 'resnet', 'c10', device); K = len(POOL)
ts = A.get_testset('c10')
CJT = os.path.join(A.CHECKPOINT_DIR, 'c10_resnet')

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

# --- Stateful embedding (penultimate features of POOL[0], the detector's encoder) ---
feat_hold = {}
def _hook(m, i, o): feat_hold['f'] = i[0].detach()
lin = None
for mod in POOL[0].modules():
    if isinstance(mod, nn.Linear): lin = mod
lin.register_forward_hook(_hook)
def embed(xt):
    with torch.no_grad(): POOL[0](xt)
    return feat_hold['f'][0].cpu().numpy()
# --- Blacklight fingerprint ---
def fingerprint(x, Lq=16, P=8):
    xq = np.clip((x*Lq).astype(np.int16), 0, Lq-1); hs = set()
    for c in range(3):
        for i in range(0, 32, P):
            for j in range(0, 32, P):
                hs.add(hash((c, i, j, xq[c, i:i+P, j:j+P].tobytes())))
    return hs

def fwd_scores(cur, xt):
    with torch.no_grad(): return POOL[cur](xt).cpu().numpy()[0]

class Defense:
    """kind in: none|rnd|aaa|stateful|blacklight|mads|mads_obfuscator"""
    def __init__(s, kind, thr=None, amax=0.0, tau=OBF_TAU):
        s.kind = kind; s.thr = thr; s.amax = amax; s.tau = tau
    def reset(s, start=0):
        s.cur = start; s.query_count = 0; s.switches = 0
        s.hist = deque(maxlen=L); s.tsw = 10**8
        s.buf = deque(maxlen=500); s.bl_idx = {}; s.bl_q = 0
    def __call__(s, x):
        xnp = x[0] if (isinstance(x, np.ndarray) and x.ndim == 4) else x
        xt = (torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x)
        if xt.dim() == 3: xt = xt[None]
        xt = xt.to(device)
        s.query_count += 1; s.tsw += 1
        if s.kind == 'rnd':
            xt = (xt + 0.03*torch.randn_like(xt)).clamp(0, 1)
            z = fwd_scores(s.cur, xt); return F.softmax(torch.tensor(z), 0).numpy()
        z = fwd_scores(s.cur, xt)
        if s.kind == 'aaa':
            c = int(z.argmax()); zo = z.copy(); zo[c] = -1e9; m2 = zo.max(); l = z[c]-m2
            lp = max(l - 2.0*np.sin(2*np.pi*l/4.0), 0.05); z2 = z.copy(); z2[c] = m2+lp
            return F.softmax(torch.tensor(z2), 0).numpy()
        if s.kind == 'none':
            return F.softmax(torch.tensor(z), 0).numpy()
        # switching defenses: decide fire
        sc_true = F.softmax(torch.tensor(z), 0).numpy(); ss = np.sort(sc_true)[::-1]
        s.hist.append(ss[0]-ss[1]); h = list(s.hist); fire = False
        if s.kind in ('mads', 'mads_obfuscator'):
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
        if s.kind == 'mads_obfuscator' and s.amax > 0:
            z = obfuscator(z, s.tsw, s.amax, s.tau if hasattr(s, 'tau') else OBF_TAU)
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

def benign_metrics(mk, bimgs, bys, nstream=10):
    corr = 0; fsw = 0
    for t in range(nstream):
        o = mk(); o.reset(0)
        for k in range(len(bimgs)):
            sc = o(bimgs[k]); corr += int(np.argmax(sc) == bys[k])
        fsw += o.switches
    return corr/(nstream*len(bimgs))*100, fsw/nstream

def calib_thr(kind, cands, bimgs):
    """pick most aggressive threshold with benign FSw/1k <= 0.5 on distinct benign stream"""
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
    bimgs = [A.img_np(ts, i)[0] for i in bidx]; bys = [A.img_np(ts, i)[1] for i in bidx]

    # calibrate detector thresholds to ~0 benign FSw (distinct benign images)
    st_thr, st_f = calib_thr('stateful', [2.0, 1.5, 1.0, 0.7, 0.5], bimgs)
    bl_thr, bl_f = calib_thr('blacklight', [0.5, 0.35, 0.25, 0.15], bimgs)
    print(f"  [calib] stateful thr={st_thr} fsw={st_f:.2f} | blacklight thr={bl_thr} fsw={bl_f:.2f}", flush=True)

    DEFS = [
        ('None (static)',   lambda: Defense('none')),
        ('RND',             lambda: Defense('rnd')),
        ('AAA',             lambda: Defense('aaa')),
        ('Stateful Det.',   lambda: Defense('stateful', thr=st_thr)),
        ('Blacklight',      lambda: Defense('blacklight', thr=bl_thr)),
        ('MADS (no obfuscator)',lambda: Defense('mads')),
        (f'MADS+obfuscator',    lambda: Defense('mads_obfuscator', amax=OBF_AMAX)),
    ]
    res = {}
    for name, mk in DEFS:
        succ = 0
        for k in range(NIMG):
            o = mk(); o.reset(0)
            succ += int(adaptive_signhunter(o, imgs[k], ys[k], MAXQ))
        asr = succ/NIMG*100
        bacc, bfsw = benign_metrics(mk, bimgs[:50], bys[:50])
        res[name] = {'adaptive_asr': round(asr, 1), 'benign_acc': round(bacc, 1), 'benign_fsw': round(bfsw, 2)}
        print(f"  {name:18s} adaptive_ASR={asr:5.1f}  benign_acc={bacc:5.1f}  fsw={bfsw:.2f}", flush=True)
        json.dump(res, open('c10_adaptive_defenses.json', 'w'), indent=2)
    print("[done]", flush=True)

if __name__ == '__main__':
    main()
