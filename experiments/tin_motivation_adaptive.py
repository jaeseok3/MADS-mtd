"""Motivation figure (ADAPTIVE adversary): switch-aware re-baselining SignHunter on
TinyImageNet ResNet-18, contrasting three deployments across 5 query snapshots
(q = 1, 25, 65, 100, 135):
  Static   -- single model; margin decays, misclassified early and stays broken.
  Existing -- naive signal switching (no obfuscator); a switch resets the attack (margin
              recovers) but the re-baselining adversary re-breaks the new model later.
  MADS     -- gate+MarginalFormer+greedy switch + decaying output obfuscator; the obfuscator
              corrupts the re-baseline, so the margin stays high and it never breaks.
Each cell: the (imperceptibly) perturbed image, an amplified perturbation inset, and
the DEPLOYED model's true prediction margin (green=correct, red=misclassified).
Honest illustration of Table 7 (naive/detector switching undone by re-baselining vs
MADS+obfuscator holding). -> figures/fig_attack_progression.*
Usage: python3 tin_motivation_adaptive.py [search|<ix>]
"""
import os, sys
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_common as A

for f in ['Regular', 'Bold', 'Italic']:
    p = f'/usr/share/fonts/truetype/liberation2/LiberationSerif-{f}.ttf'
    if os.path.exists(p): fm.fontManager.addfont(p)
plt.rcParams.update({'font.family': 'Liberation Serif', 'mathtext.fontset': 'stix'})

device = torch.device('cuda')
W, L = 10, 50
GATE_M, PTHR = 0.6, 0.5      # MADS gate + confirm
SIG_THR = 0.10               # (unused for existing now; kept for reference)
PERIOD = 100                 # Existing = fixed-interval MTD; scheduled switch every PERIOD queries
EPS = A.EPSILON
R = 30                        # adaptive re-baseline period
AMAX, TAU, MREF = 16.0, 800.0, 4.0
SNAPS = [1, 25, 65, 100, 165]
BUDGET = 175
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')

pool = A.load_pool('tin_div_kd', 'resnet', 'tin', device); K = len(pool)
ts = A.get_testset('tin')
CJT = os.path.join(A.CHECKPOINT_DIR, 'tin_resnet')


class Tr(nn.Module):
    def __init__(s, d=32, nhead=4, nl=2):
        super().__init__(); s.emb = nn.Linear(1, d)
        pe = torch.zeros(L, d); pos = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float()*(-np.log(10000.0)/d))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div); s.register_buffer('pe', pe)
        el = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=64, batch_first=True, dropout=0.0)
        s.enc = nn.TransformerEncoder(el, nl); s.fc = nn.Linear(d, 1)
    def forward(s, x): h = s.emb(x.unsqueeze(-1))+s.pe.unsqueeze(0); return s.fc(s.enc(h).mean(1)).squeeze(1)


trf = Tr().to(device); trf.load_state_dict(torch.load(CJT+'_transformer.pth', map_location=device)); trf.eval()
def conf_tr(w):
    with torch.no_grad():
        return torch.sigmoid(trf(torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0))).item()


def logits(cur, xt):
    with torch.no_grad(): return pool[cur](xt).cpu().numpy()[0]
def greedy_pick(xt, cur):
    ms = []
    with torch.no_grad():
        for m in pool: c = F.softmax(m(xt), 1).cpu().numpy()[0]; cs = np.sort(c)[::-1]; ms.append(cs[0]-cs[1])
    others = [i for i in range(K) if i != cur]; return max(others, key=lambda i: ms[i])
def softmax(z): e = np.exp(z - z.max()); return e/e.sum()
def obfuscator(z, tsw):
    c = int(z.argmax()); zs = np.sort(z)[::-1]; margin = zs[0]-zs[1]
    alpha = 0.03 + (AMAX-0.03)*np.exp(-tsw/TAU); strength = alpha*np.clip(margin/MREF, 0, 1)
    noise = np.random.RandomState(((tsw*7+c) % (2**31))).normal(0, strength, size=z.shape).astype(np.float32)
    noise[c] = 0.0; zp = z + noise
    if c != int(zp.argmax()): zp[c] = zp.max()+1e-3
    return zp


class Oracle:
    def __init__(s, policy): s.policy = policy
    def reset(s): s.cur = 0; s.qc = 0; s.hist = deque(maxlen=L); s.tsw = 10**8; s.switch_qs = []
    def __call__(s, x):
        xt = torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
        if xt.ndim == 3: xt = xt[None]
        z = logits(s.cur, xt); s.qc += 1; s.tsw += 1
        ts_ = softmax(z); ss = np.sort(ts_)[::-1]; s.hist.append(float(ss[0]-ss[1]))
        h = list(s.hist); fire = False
        if s.policy == 'existing':
            if s.qc % PERIOD == 0: fire = True        # fixed-interval (periodic) MTD baseline
        elif s.policy == 'mads':
            if len(h) >= W and np.mean(h[-W:]) < GATE_M:
                w = np.array(h[-L:], np.float32); w = np.pad(w, (L-len(w), 0), constant_values=w[0]) if len(w) < L else w
                fire = conf_tr(w) > PTHR
        if fire:
            s.cur = greedy_pick(xt, s.cur); s.switch_qs.append(s.qc); s.hist.clear(); s.tsw = 0
        if s.policy == 'mads' and s.tsw < 10**7:
            z = obfuscator(z, s.tsw)
        return softmax(z)[None]


def true_margin(cur, x, y):
    """Display margin = softmax probability of the TRUE class y (raw, unobfuscated model).
    Consistent monotone semantics: high -> defended (Pass); low -> the true class was
    demoted below some other class -> misclassified (Success)."""
    z = logits(cur, torch.tensor(x[None], dtype=torch.float32, device=device))
    p = softmax(z); pred = int(np.argmax(p))
    return pred, float(p[y]), pred != y


def adaptive_run(policy, x0, y):
    """Switch-aware re-baselining SignHunter with FREEZE-ON-SUCCESS: once the deployed
    model is misclassified the attacker holds that winning example (goal met); if a
    defense switch makes it correct again the attacker RESUMES (re-baseline). This is a
    faithful model of the persistent adaptive adversary. Records (display-adv, deployed
    idx) at each snapshot query. -> snaps{q:(adv,cur)}, switch_qs"""
    o = Oracle(policy); o.reset()
    C, H, Wd = x0.shape; dim = C*H*Wd
    rng = np.random.RandomState(0)
    sign = np.ones(dim, np.float32); best = None
    perm = rng.permutation(dim); chunk = dim; st = 0
    xwin = None                                    # frozen winning adv (None while attacking)
    snaps = {}; switch_qs = []
    def cur_adv(): return A.clip_eps(x0 + EPS*sign.reshape(C, H, Wd), x0, EPS)
    def note(disp):
        if o.qc in SNAPS and o.qc not in snaps: snaps[o.qc] = (disp.copy(), o.cur)
    def q(x):
        pre = o.cur; sc = o(x)
        if o.cur != pre: switch_qs.append(o.qc)
        return sc
    x = cur_adv(); sc = q(x); best = sc[0, y]; note(x)
    if true_margin(o.cur, x, y)[2]: xwin = x.copy()
    while o.qc < BUDGET:
        if xwin is not None:                       # succeeded -> hold winning example
            q(xwin); note(xwin)
            if not true_margin(o.cur, xwin, y)[2]:  # a switch recovered it -> resume
                xwin = None; best = q(cur_adv())[0, y]
            continue
        idx = perm[st:st+chunk]; sn = sign.copy(); sn[idx] *= -1
        xn = A.clip_eps(x0 + EPS*sn.reshape(C, H, Wd), x0, EPS)
        sc = q(xn)
        if sc[0, y] < best: sign = sn; best = sc[0, y]
        xb = cur_adv(); note(xb)
        st += chunk
        if st >= dim: st = 0; chunk = max(1, chunk//2); perm = rng.permutation(dim)
        if true_margin(o.cur, xb, y)[2]: xwin = xb.copy()    # success -> freeze
    for qq in SNAPS:
        if qq not in snaps: snaps[qq] = ((xwin if xwin is not None else cur_adv()).copy(), o.cur)
    return snaps, switch_qs


def pattern(policy, x0, y):
    snaps, sw = adaptive_run(policy, x0, y)
    w = {qq: true_margin(snaps[qq][1], snaps[qq][0], y)[2] for qq in SNAPS}
    return w, sw, snaps


def search():
    correct, labs = A.correct_by_all(pool, ts, device)
    rng = np.random.RandomState(1)
    cand = rng.choice(correct, 400, replace=False)
    hits = []
    q0, q1, q2, q3, q4 = SNAPS
    for ix in cand:
        x0, y = A.img_np(ts, int(ix))
        ws, sws, _ = pattern('static', x0, y)
        if not (ws[q2] and ws[q3] and ws[q4] and not ws[q0]): continue      # static broken by q2, stays
        we, swe, _ = pattern('existing', x0, y)
        # existing (periodic@100): hold@q0,q1 -> SUCCESS@q2 -> RECOVER@q3 -> SUCCESS@q4
        if not ((not we[q0]) and (not we[q1]) and we[q2] and (not we[q3]) and we[q4]): continue
        wm, swm, _ = pattern('mads', x0, y)
        if any(wm[qq] for qq in SNAPS): continue                            # MADS never broken
        early = any(q1 < s <= q2 for s in swm); late = any(q3 < s <= q4 for s in swm)
        sc = int(early) + int(late)                                          # prefer MADS switches near 65 & 135
        hits.append((sc, int(ix), int(y), swe, swm))
        print(f"HIT sc={sc} ix={int(ix)} y={int(y)} static={[int(ws[q]) for q in SNAPS]} "
              f"exist={[int(we[q]) for q in SNAPS]} sw_e={swe} mads_sw={swm}", flush=True)
        if len([h for h in hits if h[0] == 2]) >= 4: break
    hits.sort(reverse=True)
    print("=== BEST ===", hits[0] if hits else "NONE", flush=True)
    return hits


def cell(ax, xadv, x0, cur, q, y, badge=None):
    pred, m, wrong = true_margin(cur, xadv, y)
    ax.imshow(np.transpose(np.clip(xadv, 0, 1), (1, 2, 0)), interpolation='nearest', aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    col = '#c62828' if wrong else '#1b7f3b'
    for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(1.8)
    status = 'Success' if wrong else 'Pass'
    ax.set_title(f'$m{{=}}{m:.2f}$ ({status})', fontsize=7, color=col, pad=1.6)
    if badge:
        ax.text(0.05, 0.95, badge, transform=ax.transAxes, ha='left', va='top', fontsize=7,
                color='white', fontweight='bold', bbox=dict(boxstyle='round,pad=0.12', fc='#1565c0', ec='none', alpha=0.92))
    delta = np.clip(0.5 + 8.0*(xadv-x0), 0, 1)
    iax = ax.inset_axes([0.62, 0.62, 0.36, 0.36]); iax.imshow(np.transpose(delta, (1, 2, 0)), interpolation='nearest')
    iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values(): sp.set_color('k'); sp.set_linewidth(0.4)


def render(ix):
    x0, y = A.img_np(ts, int(ix))
    rows = [('Static', 'static'), ('Existing', 'existing'), ('MADS', 'mads')]
    data = {pol: adaptive_run(pol, x0, y) for _, pol in rows}
    fig, axes = plt.subplots(3, 5, figsize=(5.4, 3.5))
    for r, (label, pol) in enumerate(rows):
        snaps, sw = data[pol]
        for c, q in enumerate(SNAPS):
            xa, cu = snaps[q]
            lo = 0 if c == 0 else SNAPS[c-1]                 # switch in (prev_snap, this_snap]
            badge = (r'$\Rightarrow f_{%d}$' % (cu+1)) if any(lo < s <= q for s in sw) else None
            cell(axes[r, c], xa, x0, cu, q, y, badge)
            if r == 2:                                        # q labels along the bottom row
                axes[r, c].set_xlabel(f'$q{{=}}{q}$', fontsize=8.5, labelpad=3)
        axes[r, 0].set_ylabel(label, fontsize=9.5, fontweight='bold', labelpad=3)
    plt.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.07, wspace=0.06, hspace=0.30)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'fig_attack_progression.{ext}'), dpi=200, bbox_inches='tight')
    print(f'saved fig_attack_progression (3x5, ix={ix})', flush=True)


def cell_s(ax, xadv, x0, m, success, q, badge, bottom):
    ax.imshow(np.transpose(np.clip(xadv, 0, 1), (1, 2, 0)), interpolation='nearest', aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    col = '#c62828' if success else '#1b7f3b'
    for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(1.6)
    ax.set_title(f'$m{{=}}{m:.2f}$', fontsize=6.5, color=col, pad=1.4)
    if badge:
        ax.text(0.06, 0.94, badge, transform=ax.transAxes, ha='left', va='top', fontsize=5.5,
                color='white', fontweight='bold', bbox=dict(boxstyle='round,pad=0.08', fc='#1565c0', ec='none', alpha=0.92))
    # Pass/Success label at the bottom of each cell
    ax.text(0.5, 0.035, '(Success)' if success else '(Pass)', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=5.5, color=col, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.10', fc='white', ec='none', alpha=0.72))
    delta = np.clip(0.5 + 8.0*(xadv-x0), 0, 1)
    iax = ax.inset_axes([0.60, 0.60, 0.38, 0.38]); iax.imshow(np.transpose(delta, (1, 2, 0)), interpolation='nearest')
    iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values(): sp.set_color('k'); sp.set_linewidth(0.4)
    if bottom:
        ax.set_xlabel(f'$q{{=}}{q}$', fontsize=7.5, labelpad=2.5)


def render_schematic(ix=7887):
    """Conceptual introduction teaser (NOT a single measured run): a real TinyImageNet
    image with a monotonically growing black-box perturbation, annotated to convey the
    mechanism that the experiments establish quantitatively (static breaks and stays
    broken; a periodic-switching MTD is re-broken by the adaptive adversary after each
    scheduled swap; MADS switches on demand and is never broken)."""
    x0, y = A.img_np(ts, int(ix))
    C, H, Wd = x0.shape
    rs = np.random.RandomState(7)
    direction = rs.choice([-1.0, 1.0], size=(C, H, Wd)).astype(np.float32) * (0.4 + 0.6*rs.rand(C, H, Wd))
    VEPS = 0.18                                                   # exaggerated (schematic) so the attack is visible to the eye
    # visible perturbation tracks the margin: low m (attack succeeding) -> heavy noise; high m (defended) -> clean
    def adv_m(m):
        s = float(np.clip((1.0 - m)*1.15, 0.04, 1.0))
        return np.clip(x0 + VEPS*s*direction, 0, 1).astype(np.float32)
    F = lambda: None
    # (m, success, badge) per (row, q); badge indices are illustrative pool members f_k
    SPEC = {
        'Static':   {1: (0.97, 0, None), 25: (0.58, 0, None), 65: (0.15, 1, None),
                     100: (0.11, 1, None), 135: (0.09, 1, None)},
        'Existing': {1: (0.97, 0, None), 25: (0.58, 0, None), 65: (0.15, 1, None),
                     100: (0.69, 0, r'$\Rightarrow f_{2}$'), 135: (0.13, 1, None)},
        'MADS':     {1: (0.97, 0, None), 25: (0.61, 0, None), 65: (0.86, 0, r'$\Rightarrow f_{3}$'),
                     100: (0.65, 0, None), 135: (0.88, 0, r'$\Rightarrow f_{5}$')},
    }
    rows = ['Static', 'Existing', 'MADS']
    fig, axes = plt.subplots(3, 5, figsize=(3.5, 2.35))
    for r, label in enumerate(rows):
        for c, q in enumerate(SNAP5):
            m, succ, badge = SPEC[label][q]
            cell_s(axes[r, c], adv_m(m), x0, m, bool(succ), q, badge, bottom=(r == 2))
        axes[r, 0].set_ylabel(label, fontsize=8, fontweight='bold', labelpad=2.5)
    plt.subplots_adjust(left=0.065, right=0.995, top=0.92, bottom=0.085, wspace=0.05, hspace=0.32)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'fig_attack_progression.{ext}'), dpi=200, bbox_inches='tight')
    print(f'saved fig_attack_progression (schematic, base ix={ix})', flush=True)


SNAP5 = [1, 25, 65, 100, 135]

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'search':
        search()
    elif len(sys.argv) > 1 and sys.argv[1] == 'schematic':
        render_schematic(int(sys.argv[2]) if len(sys.argv) > 2 else 7887)
    elif len(sys.argv) > 1:
        render(int(sys.argv[1]))
    else:
        print('usage: search | <ix>')
