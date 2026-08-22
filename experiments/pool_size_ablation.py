"""Pool-size (K) ablation on CIFAR-100 ResNet: how many models does the defender
need in rotation? For K in {2..10} (nested subsets of the incrementally-grown K=10
Divergent KD pool; the first 5 members are identical to the K=5 pool, so K<=5 is
unchanged) measure
  cross_asr : avg off-diagonal PGD-20 transfer among the K models (pool diversity;
              pairwise, so ~flat in K -- diversity is a per-pair property)
  mads_asr  : MADS pure-attack Avg ASR (3 score attacks), greedy selection limited
              to the K deployed models -- more switch options -> lower ASR (with
              diminishing returns)
  switches  : mean switches/episode
-> pool_size_ablation.json"""
import os, sys, json, pickle
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_common as A

device = torch.device('cuda'); torch.backends.cudnn.benchmark = True
W, L, GATE = 10, 50, 0.6; EPS = 8/255; PGD_STEPS = 20
FULL = A.load_pool('divergent_kd_10_pool', 'resnet', 'cifar', device); KMAX = len(FULL)
ts = A.get_testset('cifar')
# Pin the attack-image set to the 5 DEPLOYED students (== divergent_kd_v5b_pool,
# copied byte-for-byte into the first 5 slots). correct_by_all over all 10 would
# shrink/shift the image subset and make K<=5 disagree with Table 3 / the K=5
# throughput harness (35.0). This keeps K=2..5 identical to the original run.
correct, labs = A.correct_by_all(FULL[:5], ts, device)
def img(i): x, _ = ts[int(i)]; return x.numpy() if isinstance(x, torch.Tensor) else x
ATT = {'Square': A.square_attack, 'SimBA': A.simba_dct_attack, 'SignH': A.signhunter_attack}
CJT = os.path.join(A.CHECKPOINT_DIR, 'cifar_resnet')

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

def pgd(model, x, y, eps, steps=PGD_STEPS):
    alpha = max(eps/4, 1/255)
    adv = (x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y); g = torch.autograd.grad(loss, adv)[0]
        adv = torch.min(torch.max(adv.detach()+alpha*g.sign(), x-eps), x+eps).clamp(0, 1).detach()
    return adv

def cross_asr(pool, X, Y, n, eps):
    K = len(pool); M = np.zeros((K, K))
    for src in range(K):
        Aadv = torch.cat([pgd(pool[src], X[s0:min(s0+128, n)], Y[s0:min(s0+128, n)], eps) for s0 in range(0, n, 128)])
        for tgt in range(K):
            with torch.no_grad():
                mis = sum((pool[tgt](Aadv[s0:min(s0+128, n)]).argmax(1) != Y[s0:min(s0+128, n)]).sum().item() for s0 in range(0, n, 128))
            M[src][tgt] = mis/n*100
    return float(np.mean([M[i][j] for i in range(K) for j in range(K) if i != j]))

class Oracle:
    # policy: mads (detection trigger + greedy) | periodic (every INTERVAL q, random
    # replacement) | random (prob PROB/query, random replacement) -- the last two are
    # the existing-MTD blind/random switching baselines.
    def __init__(s, pool, rng, policy='mads', interval=100, prob=0.01):
        s.pool = pool; s.K = len(pool); s.rng = rng; s.policy = policy; s.interval = interval; s.prob = prob
    def reset(s, start=0): s.cur = start % s.K; s.query_count = 0; s.switches = 0; s.hist = deque(maxlen=L); s.since = 0
    def _greedy(s, xt):
        ms = []
        with torch.no_grad():
            for m in s.pool: c = F.softmax(m(xt), 1).cpu().numpy()[0]; cs = np.sort(c)[::-1]; ms.append(cs[0]-cs[1])
        others = [i for i in range(s.K) if i != s.cur]; return max(others, key=lambda i: ms[i])
    def __call__(s, x):
        xt = (torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x).to(device)
        with torch.no_grad(): sc = F.softmax(s.pool[s.cur](xt), 1).cpu().numpy()[0]
        ss = np.sort(sc)[::-1]; s.hist.append(ss[0]-ss[1]); s.query_count += 1; s.since += 1
        h = list(s.hist); fire = False; greedy = False
        if s.policy == 'periodic': fire = s.since >= s.interval
        elif s.policy == 'random': fire = s.rng.random() < s.prob
        elif s.policy == 'mads':
            greedy = True
            if len(h) >= W and np.mean(h[-W:]) < GATE and s.K > 1:
                w = np.array(h[-L:]); w = np.pad(w, (L-len(w), 0), constant_values=w[0]) if len(w) < L else w
                fire = conf_tr(w) > 0.5
        if fire and s.K > 1:
            others = [i for i in range(s.K) if i != s.cur]
            s.cur = s._greedy(xt) if greedy else int(s.rng.choice(others))
            s.switches += 1; s.since = 0; s.hist.clear()
        return sc[None]

def policy_asr(subpool, idxs, policy):
    tot = 0; sw = 0; ep = 0
    for an, af in ATT.items():
        succ = 0
        for k, idx in enumerate(idxs):
            srng = np.random.RandomState(3000+k); o = Oracle(subpool, srng, policy=policy); o.reset(int(srng.randint(0, len(subpool))))
            np.random.seed(4000+k)
            _, _, s = af(o, img(idx), int(labs[idx]), 1000); succ += int(s); sw += o.switches; ep += 1
        tot += succ
    return tot/(len(idxs)*len(ATT))*100, sw/ep

def main():
    glob = np.random.RandomState(7)
    # draw MADS attack targets FIRST so they match the Table-3/throughput harness
    # (glob state), then the cross-ASR image set.
    idxs = glob.choice(correct, 60, replace=False)  # attack targets
    cidx = glob.choice(correct, 200, replace=False)
    X = torch.stack([torch.from_numpy(img(i)).float() for i in cidx]).to(device); Y = torch.tensor([int(labs[i]) for i in cidx], device=device)
    res = {}
    for K in range(2, KMAX+1):
        sub = FULL[:K]
        ca = cross_asr(sub, X, Y, 200, EPS)
        am, swm = policy_asr(sub, idxs, 'mads')       # MADS (detection + greedy)
        ap, swp = policy_asr(sub, idxs, 'periodic')   # existing MTD: periodic (100q), random replacement
        ar, swr = policy_asr(sub, idxs, 'random')     # existing MTD: random (p=0.01), random replacement
        res[str(K)] = {'cross_asr': round(ca, 1), 'mads_asr': round(am, 1), 'switches': round(swm, 2),
                       'periodic_asr': round(ap, 1), 'random_asr': round(ar, 1)}
        print(f"  K={K}  cross_ASR={ca:.1f}  MADS={am:.1f}(sw{swm:.2f})  periodic={ap:.1f}(sw{swp:.2f})  random={ar:.1f}(sw{swr:.2f})", flush=True)
        json.dump(res, open('pool_size_ablation.json', 'w'), indent=2)
    print("[done]", flush=True)

if __name__ == '__main__':
    main()
