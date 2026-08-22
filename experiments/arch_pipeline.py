"""Unified architecture/dataset evaluation harness. One script, four tasks:
  confirm  : harvest margin windows on a pool, train Transformer + GBM confirmers
  ch2      : Challenge-2 Settings 1/2/3 (six policies)
  nonmtd   : non-MTD defense comparison (250-query burst)
  hardlabel: HSJA / CGBA hard-label attacks vs static / AAA / detectors / ours
Works for any (arch in {resnet,vgg,alexnet}) x (dataset in {cifar,tin}) via arch_common.
Confirmers/results are written with a --tag prefix so runs never collide."""
import os, sys, json, time, argparse, pickle
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque, Counter
from sklearn.ensemble import GradientBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_common as A

W = 10; L = 50; GATE = 0.6; EPS = A.EPSILON
device = torch.device('cuda'); torch.backends.cudnn.benchmark = True


class Tr(nn.Module):
    def __init__(s, d=32, nhead=4, nl=2):
        super().__init__(); s.emb = nn.Linear(1, d)
        pe = torch.zeros(L, d); pos = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float()*(-np.log(10000.0)/d))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div); s.register_buffer('pe', pe)
        el = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=64, batch_first=True, dropout=0.0)
        s.enc = nn.TransformerEncoder(el, nl); s.fc = nn.Linear(d, 1)
    def forward(s, x): h = s.emb(x.unsqueeze(-1))+s.pe.unsqueeze(0); return s.fc(s.enc(h).mean(1)).squeeze(1)


def feats(M):  # M: [n,L]
    return np.stack([M.mean(1), M.min(1), M[:, -1], (M < 0.3).mean(1),
                     np.array([np.polyfit(np.arange(L), r, 1)[0]*L for r in M]), M.std(1)], 1).astype(np.float32)

def windows(ms, gate=True):
    out = []
    for t in range(W-1, len(ms)):
        if t < L-1:
            w = np.array(ms[:t+1]); w = np.pad(w, (L-len(w), 0), constant_values=w[0] if len(w) else 1.0)
        else: w = np.array(ms[t-L+1:t+1])
        if (not gate) or np.mean(w[-W:]) < GATE: out.append(w.astype(np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["confirm", "ch2", "nonmtd", "hardlabel"])
    ap.add_argument("--arch", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--pool", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--fast", type=int, default=1)
    ap.add_argument("--hl_budget", type=int, default=2000)
    ap.add_argument("--ml_thr", type=float, default=0.5)  # single-stage ML operating point
    ap.add_argument("--burst_q", type=int, default=250)   # nonmtd attack budget
    ap.add_argument("--nonmtd_prefix", type=int, default=1)  # 1=benign prefix (detectors), 0=pure attack
    ap.add_argument("--nn", type=int, default=0)          # nonmtd episodes/attack (0=use fast default)
    ap.add_argument("--out_suffix", type=str, default="")  # append to output json tag
    ap.add_argument("--detectors", type=str, default="stateful,blacklight,ours")  # nonmtd detector subset
    ap.add_argument("--skip_output", type=int, default=0)  # 1=skip static/rnd/aaa (already measured)
    ap.add_argument("--gate", type=float, default=0.6)  # loose recall gate; calibrate to benign margin scale (10-class CIFAR-10 needs ~0.85)
    ap.add_argument("--pthr", type=float, default=0.5)  # MarginalFormer confirm threshold (raise to cut benign false switches)
    ap.add_argument("--sig_thr", type=float, default=0.3)  # raw Signal detector margin threshold (setting1/3)
    ap.add_argument("--ops_curve", type=int, default=0)  # 1=report ASR/fsw at every 'ours' threshold (operating-point trade-off)
    ap.add_argument("--hl_defs", type=str, default="")  # hardlabel: restrict defenses (e.g. static,aaa,ours); empty=all
    args = ap.parse_args()
    global GATE; GATE = args.gate
    pool = A.load_pool(args.pool, args.arch, args.dataset, device); K = len(pool)
    testset = A.get_testset(args.dataset)
    correct, labs = A.correct_by_all(pool, testset, device)
    print(f"[{args.task}] arch={args.arch} ds={args.dataset} pool={args.pool} K={K} correct={len(correct)}", flush=True)
    def img(i): x, _ = testset[int(i)]; return (x.numpy() if isinstance(x, torch.Tensor) else x)
    CJT = os.path.join(A.CHECKPOINT_DIR, args.tag)
    ATT = {'Square': A.square_attack, 'SimBA': A.simba_dct_attack, 'SignH': A.signhunter_attack}

    # ---------------- CONFIRM ----------------
    if args.task == "confirm":
        rng = np.random.RandomState(0); N = len(testset)
        bw = []; bw_all = []
        for _ in range(80):
            ms = []
            for i in rng.randint(0, N, size=300):
                x = img(i); xt = torch.from_numpy(x).float().to(device)
                with torch.no_grad(): sc = F.softmax(pool[0](xt[None] if x.ndim == 3 else xt), 1).cpu().numpy()[0]
                ss = np.sort(sc)[::-1]; ms.append(ss[0]-ss[1])
            bw += windows(ms); bw_all += windows(ms, gate=False)
        aw = []
        class Rec:
            def __init__(s): s.query_count = 0; s.ms = []
            def reset(s, st=0): s.query_count = 0
            def __call__(s, xx):
                xt = torch.from_numpy(xx).float().to(device)
                with torch.no_grad(): sc = F.softmax(pool[0](xt), 1).cpu().numpy()
                s.query_count += xx.shape[0]; ss = np.sort(sc[0])[::-1]; s.ms.append(ss[0]-ss[1]); return sc
        for ix in rng.choice(correct, 200, replace=False):
            o = Rec(); A.signhunter_attack(o, img(ix), int(labs[ix]), 300); aw += windows(o.ms)
        Xb = np.array(bw).reshape(-1, L); Xa = np.array(aw).reshape(-1, L)
        if len(Xb) < 50:
            raise SystemExit(f"[confirm] only {len(Xb)} gated benign windows at GATE={GATE}; raise --gate for this margin scale")
        X = np.concatenate([Xb, Xa]); Y = np.concatenate([np.zeros(len(Xb)), np.ones(len(Xa))]).astype(np.float32)
        p = rng.permutation(len(X)); X, Y = X[p], Y[p]
        print(f"  benign(gated)={len(Xb)} benign(all)={len(bw_all)} attack={len(Xa)}", flush=True)
        # ML Filter = single-stage detector run on EVERY window (no static gate); its GBM is
        # therefore trained on the FULL benign window distribution (bw_all), not just low-margin.
        Xb_all = np.array(bw_all)
        Xg = np.concatenate([Xb_all, Xa]); Yg = np.concatenate([np.zeros(len(Xb_all)), np.ones(len(Xa))]).astype(np.float32)
        pg = rng.permutation(len(Xg)); Xg, Yg = Xg[pg], Yg[pg]
        gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05).fit(feats(Xg), Yg)
        pickle.dump(gbm, open(CJT+"_gbm.pkl", "wb"))
        # MarginalFormer stays the two-stage confirmer, trained on gated (low-margin) windows.
        Xt = torch.tensor(X); Yt = torch.tensor(Y); trf = Tr(); opt = torch.optim.Adam(trf.parameters(), 1e-3)
        idx = np.arange(len(X))
        for ep in range(40):
            np.random.shuffle(idx)
            for b in range(0, len(X), 256):
                bi = idx[b:b+256]; lo = trf(Xt[bi]); loss = F.binary_cross_entropy_with_logits(lo, Yt[bi])
                opt.zero_grad(); loss.backward(); opt.step()
        torch.save(trf.state_dict(), CJT+"_transformer.pth")
        print("[done] saved confirmers", flush=True); return

    # confirmers for the eval tasks
    gbm = pickle.load(open(CJT+"_gbm.pkl", "rb"))
    trf = Tr().to(device); trf.load_state_dict(torch.load(CJT+"_transformer.pth", map_location=device)); trf.eval()
    def conf_tr(w):
        with torch.no_grad(): return torch.sigmoid(trf(torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0))).item()
    def conf_ml(w): return float(gbm.predict_proba(feats(w[None]))[0, 1])
    def greedy(xt):
        ms = []
        with torch.no_grad():
            for m in pool: c = F.softmax(m(xt), 1).cpu().numpy()[0]; cs = np.sort(c)[::-1]; ms.append(cs[0]-cs[1])
        return ms

    class Oracle:
        def __init__(s, policy, rng, thr=0.5, interval=100, prob=0.01, mthr=0.3):
            s.policy = policy; s.rng = rng; s.thr = thr; s.interval = interval; s.prob = prob; s.mthr = mthr
        def reset(s, start=0): s.cur = start; s.query_count = 0; s.switches = 0; s.hist = deque(maxlen=L); s.since = 0
        def _sw(s, xt, g):
            others = [i for i in range(K) if i != s.cur]
            s.cur = max(others, key=lambda i: greedy(xt)[i]) if g else s.rng.choice(others)
            s.switches += 1; s.since = 0; s.hist.clear()
        def __call__(s, x):
            xt = (torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x).to(device)
            with torch.no_grad(): sc = F.softmax(pool[s.cur](xt), 1).cpu().numpy()[0]
            ss = np.sort(sc)[::-1]; s.hist.append(ss[0]-ss[1]); s.query_count += 1; s.since += 1
            p = s.policy; fire = False; g = False; h = list(s.hist)
            if p == 'periodic': fire = s.since >= s.interval
            elif p == 'random': fire = s.rng.random() < s.prob
            elif p == 'signal': fire = len(h) >= W and np.mean(h[-W:]) < s.mthr
            elif p == 'ml':          # single-stage: GBM inspects EVERY window (no gate); random replacement (greedy=MADS-only)
                g = False
                if len(h) >= W:
                    w = np.array(h[-L:]); w = np.pad(w, (L-len(w), 0), constant_values=w[0]) if len(w) < L else w
                    fire = conf_ml(w) > s.thr
            elif p == 'transformer':  # two-stage: static gate (recall) + Transformer (precision)
                g = True
                if len(h) >= W and np.mean(h[-W:]) < GATE:
                    w = np.array(h[-L:]); w = np.pad(w, (L-len(w), 0), constant_values=w[0]) if len(w) < L else w
                    fire = conf_tr(w) > s.thr
            if fire: s._sw(xt, g)
            return sc[None]

    POL = [('static', {}), ('periodic', {'interval': 100}), ('random', {'prob': 0.01}),
           ('signal', {'mthr': args.sig_thr}), ('ml', {'thr': args.ml_thr}), ('transformer', {'thr': args.pthr})]
    _only = os.environ.get('POL_ONLY', '')
    if _only:
        _keep = set(_only.split(',')); POL = [p for p in POL if p[0] in _keep]  # fast: restrict evaluated policies
    N = len(testset); glob = np.random.RandomState(7)

    # ---------------- CH2 ----------------
    if args.task == "ch2":
        out = {}
        NS1 = 100 if args.fast else 200; NS2 = 60 if args.fast else 120; NS2F = 30 if args.fast else 40; NS3 = 12 if args.fast else 24
        idxs = glob.choice(correct, NS1, replace=False); s1 = {}
        for pn, kw in POL:
            row = {}; sw = 0; sn = 0
            for an, af in ATT.items():
                succ = 0
                for k, idx in enumerate(idxs):
                    srng = np.random.RandomState(3000+k); o = Oracle(pn, srng, **kw); o.reset(int(srng.randint(0, K)))
                    np.random.seed(4000+k)
                    _, _, s = af(o, img(idx), int(labs[idx]), 1000); succ += int(s); sw += o.switches; sn += 1
                row[an] = succ/NS1*100
            row['Avg'] = float(np.mean([row[a] for a in ATT])); row['Sw'] = sw/sn; s1[pn] = row
            print(f"[S1 {pn:11s}] Sq{row['Square']:.1f} Si{row['SimBA']:.1f} SH{row['SignH']:.1f} Avg{row['Avg']:.1f} Sw{row['Sw']:.2f}", flush=True)
        out['setting1'] = s1; json.dump(out, open(f'arch_ch2_{args.tag}{args.out_suffix}.json', 'w'), indent=2)
        tg = glob.choice(correct, 300, replace=False); benign = [img(i) for i in glob.randint(0, N, size=2000)]; BURST = 250; STREAM = 1000
        def bfsw(pn, kw):
            tot = 0
            for b in range(NS2F):
                srng = np.random.RandomState(7000+b); o = Oracle(pn, srng, **kw); o.reset(0)
                for _ in range(1000): o(benign[srng.randint(0, len(benign))][None])
                tot += o.switches
            return tot/NS2F/1000*1000
        def basr(pn, kw):
            succ = 0
            for b in range(NS2):
                srng = np.random.RandomState(5000+b); o = Oracle(pn, srng, **kw); o.reset(0); cp = srng.randint(0, STREAM-BURST)
                for _ in range(cp): o(benign[srng.randint(0, len(benign))][None])
                np.random.seed(9000+b); idx = int(srng.choice(tg))
                _, _, s = A.signhunter_attack(o, img(idx), int(labs[idx]), o.query_count+min(BURST, STREAM-o.query_count)); succ += int(s)
            return succ/NS2*100
        op = {'static': [('-', {})], 'periodic': [('-', {'interval': 100})], 'random': [('-', {'prob': 0.01})],
              'signal': [('perf', {'mthr': 0.3}), ('evasion', {'mthr': 0.5})],
              'ml': [('perf', {'thr': 0.5}), ('evasion', {'thr': 0.15})],
              'transformer': [('perf', {'thr': args.pthr}), ('evasion', {'thr': 0.15})]}
        S2 = {}
        for pn, _ in POL:
            S2[pn] = {}
            for opn, kw in op[pn]:
                S2[pn][opn] = [bfsw(pn, kw), basr(pn, kw)]
                print(f"[S2 {pn:11s} {opn:7s}] fsw={S2[pn][opn][0]:.2f} burst={S2[pn][opn][1]:.1f}", flush=True)
        out['setting2'] = S2; json.dump(out, open(f'arch_ch2_{args.tag}{args.out_suffix}.json', 'w'), indent=2)
        benign4 = [img(i) for i in glob.randint(0, N, size=4000)]; MIL = [1000, 4000, 16000]; S3 = {}
        for pn, kw in POL:
            cur = np.zeros(3)
            for b in range(NS3):
                srng = np.random.RandomState(7000+b); o = Oracle(pn, srng, **kw); o.reset(0); mi = 0
                for q in range(1, 16001):
                    o(benign4[srng.randint(0, len(benign4))][None])
                    if mi < 3 and q == MIL[mi]: cur[mi] += o.switches; mi += 1
            S3[pn] = (cur/NS3).tolist(); print(f"[S3 {pn:11s}] {[round(v,2) for v in S3[pn]]}", flush=True)
        out['setting3'] = S3; json.dump(out, open(f'arch_ch2_{args.tag}{args.out_suffix}.json', 'w'), indent=2)
        print("[done]", flush=True); return

    # shared: single-model defenses + detector triggers for nonmtd / hardlabel
    model = pool[0]
    fh = {}
    for mod in model.modules():
        if isinstance(mod, nn.Linear): _lin = mod
    _lin.register_forward_hook(lambda m, i, o: fh.__setitem__('f', i[0].detach()))
    def embed(xt):
        with torch.no_grad(): model(xt)
        return fh['f'][0].cpu().numpy()
    HW = testset[0][0].shape[-1]
    def fingerprint(x, Lq=16, P=8):
        xq = np.clip((x*Lq).astype(np.int16), 0, Lq-1); hs = set()
        for c in range(3):
            for i in range(0, HW, P):
                for j in range(0, HW, P): hs.add(hash((c, i, j, xq[c, i:i+P, j:j+P].tobytes())))
        return hs
    def gsw(o, xt):
        others = [i for i in range(K) if i != o.cur]; o.cur = max(others, key=lambda i: greedy(xt)[i]); o.switches += 1
    class RND:
        def __init__(s, sigma=0.03): s.sigma = sigma
        def reset(s, st=0): s.cur = 0; s.query_count = 0; s.switches = 0
        def __call__(s, x):
            xt = torch.from_numpy(x).float(); xt = (xt + s.sigma*torch.randn_like(xt)).clamp(0, 1).to(device); s.query_count += 1
            with torch.no_grad(): return F.softmax(model(xt), 1).cpu().numpy()
    class AAA:
        def __init__(s, a=2.0, T=4.0): s.a = a; s.T = T
        def reset(s, st=0): s.cur = 0; s.query_count = 0; s.switches = 0
        def __call__(s, x):
            xt = torch.from_numpy(x).float().to(device)
            with torch.no_grad(): z = model(xt)[0].cpu().numpy()
            s.query_count += 1; c = int(z.argmax()); zo = z.copy(); zo[c] = -1e9; m2 = zo.max(); l = z[c]-m2
            lp = max(l - s.a*np.sin(2*np.pi*l/s.T), 0.05); z2 = z.copy(); z2[c] = m2+lp
            return F.softmax(torch.tensor(z2)[None], 1).numpy()
    class Static:
        def reset(s, st=0): s.cur = 0; s.query_count = 0; s.switches = 0
        def __call__(s, x):
            xt = torch.from_numpy(x).float().to(device)
            with torch.no_grad(): sc = F.softmax(model(xt), 1).cpu().numpy()
            s.query_count += 1; return sc
    WINBL = 1500
    class Trig:  # ours / stateful / blacklight
        def __init__(s, kind, thr): s.kind = kind; s.thr = thr
        def reset(s, st=0): s.cur = 0; s.query_count = 0; s.switches = 0; s.hist = deque(maxlen=L); s.buf = deque(maxlen=500); s.win = deque(maxlen=WINBL); s.cnt = {}
        def __call__(s, x):
            xnp = x[0] if x.ndim == 4 else x; xt = torch.from_numpy(x).float().to(device)
            with torch.no_grad(): sc = F.softmax(pool[s.cur](xt), 1).cpu().numpy()[0]
            ss = np.sort(sc)[::-1]; s.hist.append(ss[0]-ss[1]); s.query_count += 1; h = list(s.hist); fire = False
            if s.kind == 'ours':
                if len(h) >= W and np.mean(h[-W:]) < GATE:
                    w = np.array(h[-L:]); w = np.pad(w, (L-len(w), 0), constant_values=w[0]) if len(w) < L else w
                    fire = conf_tr(w) > s.thr
            elif s.kind == 'stateful':
                e = embed(xt)
                if len(s.buf) >= 10:
                    B = np.stack(s.buf); dk = np.sort(np.linalg.norm(B-e, axis=1))[:min(5, len(B))].mean()
                    if dk < s.thr: fire = True
                s.buf.append(e)
            elif s.kind == 'blacklight':
                fp = fingerprint(xnp); ov = sum(1 for hh in fp if s.cnt.get(hh, 0) > 0)
                fire = ov/len(fp) >= s.thr
                if len(s.win) == WINBL:
                    for hh in s.win[0]:
                        s.cnt[hh] -= 1
                        if s.cnt[hh] <= 0: del s.cnt[hh]
                s.win.append(fp)
                for hh in fp: s.cnt[hh] = s.cnt.get(hh, 0)+1
            if fire: gsw(s, xt); s.hist.clear()
            return sc[None]

    tg = glob.choice(correct, 300, replace=False)

    # ---------------- NONMTD (250-query burst) ----------------
    if args.task == "nonmtd":
        BURST = args.burst_q; Nn = args.nn if args.nn > 0 else (40 if args.fast else 60)
        benign = [img(i) for i in glob.randint(0, N, size=20000)]
        res = {}
        def burst_single(mk):
            out = {}
            for an, af in ATT.items():
                succ = 0
                for b in range(Nn):
                    srng = np.random.RandomState(5000+b); o = mk(); o.reset(); np.random.seed(9000+b); idx = int(srng.choice(tg))
                    _, _, s = af(o, img(idx), int(labs[idx]), BURST); succ += int(s)
                out[an] = succ/Nn*100
            out['Avg'] = float(np.mean([out[a] for a in ATT])); return out
        def clean_acc(mk, n=2000):
            d = mk(); d.reset(); ix = glob.choice(N, n, replace=False); ok = 0
            for i in ix:
                x, y = testset[int(i)]; x = (x.numpy() if isinstance(x, torch.Tensor) else x)
                ok += int(d(x[None])[0].argmax() == y)
            return ok/n*100
        out_json = f'arch_nonmtd_{args.tag}{args.out_suffix}.json'
        if os.path.exists(out_json): res = json.load(open(out_json))  # resume/merge prior rows
        if not args.skip_output:
            for nm, mk in [('static', lambda: Static()), ('rnd', lambda: RND(0.03)), ('aaa', lambda: AAA())]:
                b = burst_single(mk); res[nm] = {**b, 'clean_acc': clean_acc(mk)}
                print(f"  {nm:10s} Sq{b['Square']:.0f} Si{b['SimBA']:.0f} SH{b['SignH']:.0f} Avg{b['Avg']:.1f} acc{res[nm]['clean_acc']:.1f}", flush=True)
                json.dump(res, open(out_json, 'w'), indent=2)
        STREAM = max(1000, BURST + 250)
        def bfsw_d(kind, thr, ns=15):
            tot = 0
            for b in range(ns):
                srng = np.random.RandomState(7000+b); o = Trig(kind, thr); o.reset()
                for i in srng.permutation(len(benign))[:1000]: o(benign[int(i)][None])
                tot += o.switches
            return tot/ns/1000*1000
        def basr_d(kind, thr):
            out = {}
            for an, af in ATT.items():
                succ = 0
                for b in range(Nn):
                    srng = np.random.RandomState(5000+b); o = Trig(kind, thr); o.reset(); perm = srng.permutation(len(benign)); cp = srng.randint(0, STREAM-BURST) if args.nonmtd_prefix else 0
                    for t in range(cp): o(benign[int(perm[t])][None])
                    np.random.seed(9000+b); idx = int(srng.choice(tg))
                    _, _, s = af(o, img(idx), int(labs[idx]), o.query_count+min(BURST, STREAM-o.query_count)); succ += int(s)
                out[an] = succ/Nn*100
            out['Avg'] = float(np.mean([out[a] for a in ATT])); return out
        CAND = {'stateful': [3.0, 2.5, 2.0, 1.5, 1.0], 'blacklight': [0.95, 0.85, 0.7, 0.55], 'ours': [0.04, 0.08, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]}
        for kind in args.detectors.split(','):
            if not kind: continue
            pick = None
            for thr in CAND[kind]:
                f = bfsw_d(kind, thr); pick = (thr, f)
                if f <= 0.15: break
            thr, f = pick; b = basr_d(kind, thr); res[kind] = {**b, 'fsw': f, 'thr': thr}
            print(f"  {kind:10s} thr={thr} fsw={f:.2f} Sq{b['Square']:.0f} Si{b['SimBA']:.0f} SH{b['SignH']:.0f} Avg{b['Avg']:.1f}", flush=True)
            # operating-point trade-off curve for our detector (ASR & fsw at every threshold)
            if kind == 'ours' and args.ops_curve:
                curve = []
                for t in CAND['ours']:
                    ff = bfsw_d('ours', t); bb = basr_d('ours', t)
                    curve.append({'thr': t, 'fsw': round(ff, 2), 'Avg': round(bb['Avg'], 1)})
                    print(f"    [ops] thr={t} fsw={ff:.2f} Avg={bb['Avg']:.1f}", flush=True)
                res['ours_curve'] = curve
            json.dump(res, open(out_json, 'w'), indent=2)
        print("[done]", flush=True); return

    # ---------------- HARDLABEL (HSJA/CGBA) ----------------
    if args.task == "hardlabel":
        mq = args.hl_budget
        def clip(x, x0, e): return np.clip(np.clip(x, x0-e, x0+e), 0, 1)
        def _bs(o, y, x0, xa, tol=2e-3):
            lo, hi = 0.0, 1.0
            while hi-lo > tol:
                m = (lo+hi)/2; itp = (1-m)*x0+m*xa
                if np.argmax(o(itp[None])) != y: hi = m
                else: lo = m
            return (1-hi)*x0+hi*xa
        def _init(o, y, x0):
            for _ in range(40):
                if o.query_count >= mq: return None
                c = np.random.rand(*x0.shape).astype(np.float32)
                if np.argmax(o(c[None])) != y: return c
            return None
        def hsja(o, x0, y):
            xa = _init(o, y, x0)
            if xa is None: return x0, False
            xa = _bs(o, y, x0, xa); it = 0
            while o.query_count < mq-60:
                dist = np.linalg.norm(xa-x0); delta = dist/np.sqrt(x0.size)*0.2; B = min(40, mq-o.query_count-2)
                if B < 2: break
                rv = np.random.randn(B, *x0.shape).astype(np.float32); rv /= (np.linalg.norm(rv.reshape(B, -1), axis=1).reshape(B, 1, 1, 1)+1e-12)
                phi = np.array([1.0 if np.argmax(o(np.clip(xa+delta*rv[b], 0, 1)[None])) != y else -1.0 for b in range(B)])
                g = phi.mean()*rv.mean(0) if abs(phi.mean()) == 1 else ((phi-phi.mean()).reshape(B, 1, 1, 1)*rv).mean(0)
                gn = g/(np.linalg.norm(g)+1e-12); t = dist/np.sqrt(it+1); xn = np.clip(xa+t*gn, 0, 1)
                while np.argmax(o(xn[None])) == y and t > 1e-3 and o.query_count < mq: t /= 2; xn = np.clip(xa+t*gn, 0, 1)
                xa = _bs(o, y, x0, xn); it += 1
            adv = clip(xa, x0, EPS); return adv, bool(np.argmax(o(adv[None])) != y)
        def cgba(o, x0, y):
            xa = _init(o, y, x0)
            if xa is None: return x0, False
            xa = _bs(o, y, x0, xa)
            while o.query_count < mq-30:
                d = (x0-xa); nd = np.linalg.norm(d)
                if nd < 1e-6: break
                d = d/nd; v = np.random.randn(*x0.shape).astype(np.float32); v -= (v*d).sum()*d; v /= (np.linalg.norm(v)+1e-12)
                r = nd; best = xa; bestd = nd; lo, hi = 0.0, np.pi/2
                for _ in range(8):
                    if o.query_count >= mq: break
                    t = (lo+hi)/2; cand = np.clip(x0-r*(np.cos(t)*d+np.sin(t)*v), 0, 1)
                    if np.argmax(o(cand[None])) != y:
                        cb = _bs(o, y, x0, cand); db = np.linalg.norm(cb-x0)
                        if db < bestd: bestd = db; best = cb
                        hi = t
                    else: lo = t
                if bestd < nd-1e-5: xa = best
                else: break
            adv = clip(xa, x0, EPS); return adv, bool(np.argmax(o(adv[None])) != y)
        idxs = np.random.RandomState(7).choice(correct, 30, replace=False)
        defs = {'static': lambda: Static(), 'rnd': lambda: RND(0.03), 'aaa': lambda: AAA(),
                'stateful': lambda: Trig('stateful', 3.0), 'blacklight': lambda: Trig('blacklight', 0.55),
                'ours': lambda: Trig('ours', 0.04)}
        if args.hl_defs:
            keep = set(args.hl_defs.split(',')); defs = {k: v for k, v in defs.items() if k in keep}
        res = {}
        for an, af in [('HSJA', hsja), ('CGBA', cgba)]:
            for dn, mk in defs.items():
                succ = 0; sw = 0
                for k, idx in enumerate(idxs):
                    np.random.seed(1000+k); o = mk(); o.reset()
                    adv, s = af(o, img(idx), int(labs[idx])); succ += int(s); sw += o.switches
                res[f'{an}_{dn}'] = {'asr': succ/len(idxs)*100, 'sw': sw/len(idxs)}
                print(f"  {an} {dn:10s} ASR={succ/len(idxs)*100:.0f}% sw={sw/len(idxs):.1f}", flush=True)
                json.dump(res, open(f'arch_hardlabel_{args.tag}{args.out_suffix}.json', 'w'), indent=2)
        print("[done]", flush=True); return


if __name__ == "__main__":
    main()
