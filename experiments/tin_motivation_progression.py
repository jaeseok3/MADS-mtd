"""Motivation figure: query-based attack (SignHunter) progression on TinyImageNet
ResNet-18, contrasting a STATIC deployment (attack accumulates -> succeeds) vs
MADS (margin drops -> confirmed switch invalidates progress -> attack never
reaches success, loops). Two rows x three query snapshots; each cell shows the
(imperceptibly) perturbed image, an amplified perturbation patch, and the
deployed model's prediction + confidence margin. -> figures/fig_attack_progression.*
"""
import os, sys
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import deque
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrowPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_common as A

for f in ['Regular', 'Bold', 'Italic']:
    p = f'/usr/share/fonts/truetype/liberation2/LiberationSerif-{f}.ttf'
    if os.path.exists(p): fm.fontManager.addfont(p)
plt.rcParams.update({'font.family': 'Liberation Serif', 'mathtext.fontset': 'stix'})

device = torch.device('cuda')
W, L, GATE, PTHR, EPS = 10, 50, 0.6, 0.5, A.EPSILON
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
def greedy_margins(xt):
    ms = []
    with torch.no_grad():
        for m in pool:
            c = F.softmax(m(xt), 1).cpu().numpy()[0]; cs = np.sort(c)[::-1]; ms.append(cs[0]-cs[1])
    return ms


class Oracle:
    def __init__(s, mads): s.mads = mads
    def reset(s):
        s.cur = 0; s.query_count = 0; s.hist = deque(maxlen=L); s.switch_qs = []
    def __call__(s, x):
        xt = torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
        with torch.no_grad(): sc = F.softmax(pool[s.cur](xt), 1).cpu().numpy()
        ss = np.sort(sc[0])[::-1]; s.hist.append(float(ss[0]-ss[1])); s.query_count += 1
        if s.mads and len(s.hist) >= W and np.mean(list(s.hist)[-W:]) < GATE:
            w = np.array(list(s.hist)[-L:], np.float32)
            w = np.pad(w, (L-len(w), 0), constant_values=w[0]) if len(w) < L else w
            if conf_tr(w) > PTHR:
                others = [i for i in range(K) if i != s.cur]; gm = greedy_margins(xt[:1] if xt.ndim == 4 else xt)
                s.cur = max(others, key=lambda i: gm[i]); s.switch_qs.append(s.query_count); s.hist.clear()
        return sc


def margin_pred(cur, x, y):
    with torch.no_grad():
        sc = F.softmax(pool[cur](torch.tensor(x[None], dtype=torch.float32, device=device)), 1).cpu().numpy()[0]
    pred = int(np.argmax(sc)); ss = np.sort(sc)[::-1]
    return pred, float(ss[0]-ss[1]), pred != y


def run(mads, x0, y, budget, snap_qs):
    """SignHunter with best-adv tracking; capture best-adv at each query in snap_qs."""
    o = Oracle(mads); o.reset()
    C, H, Wd = x0.shape; dim = C*H*Wd
    sign = np.ones(dim, np.float32)
    xadv = A.clip_eps(x0 + EPS*sign.reshape(C, H, Wd), x0, EPS)
    snaps = {}; succ_q = None
    def snap():
        if o.query_count in snap_qs and o.query_count not in snaps:
            snaps[o.query_count] = (xadv.copy(), o.cur)
    sc = o(xadv[None]); snap()
    if np.argmax(sc) != y: succ_q = o.query_count
    best = sc[0, y]; perm = np.random.RandomState(0).permutation(dim); chunk = dim
    while o.query_count < budget and chunk >= 1 and succ_q is None:
        for st in range(0, dim, chunk):
            if o.query_count >= budget or succ_q is not None: break
            idx = perm[st:st+chunk]; sn = sign.copy(); sn[idx] *= -1
            xn = A.clip_eps(x0 + EPS*sn.reshape(C, H, Wd), x0, EPS)
            sc = o(xn)
            if sc[0, y] < best:
                sign = sn; best = sc[0, y]; xadv = xn
            snap()
            if np.argmax(sc) != y: succ_q = o.query_count; break
        chunk //= 2
    # ensure the last snapshot exists even if loop ended early
    for q in snap_qs:
        if q not in snaps: snaps[q] = (xadv.copy(), o.cur)
    return snaps, succ_q, o.switch_qs


# ---- pick an illustrative image: static succeeds in a small budget, MADS switches
#      before that and is NOT broken by then ----
ix = 7281   # chosen illustrative image (static succeeds ~q65, MADS switches ~q25)
x0, y = A.img_np(ts, ix)
_, s_q, _ = run(False, x0, y, 90, set())
_, _mq, m_sw = run(True, x0, y, s_q, set())
qs = [1, max(2, m_sw[0]), s_q]      # snapshot at start, at the MADS switch, at static success
print(f"chosen img {ix}  true={y}  static_success_q={s_q}  MADS_switches={m_sw}  snaps={qs}", flush=True)

x0, _ = A.img_np(ts, ix)
s_snaps, s_q, _ = run(False, x0, y, s_q, set(qs))
m_snaps, m_q, m_sw = run(True, x0, y, s_q, set(qs))
print(f"MADS success within {s_q}q? {m_q}", flush=True)


def cell(ax, xadv, cur, q, is_switch=False):
    pred, m, wrong = margin_pred(cur, xadv, y)
    ax.imshow(np.transpose(np.clip(xadv, 0, 1), (1, 2, 0)), interpolation='nearest', aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    col = '#c62828' if wrong else '#2e7d32'
    for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(1.6)
    ax.set_title(f'$q{{=}}{q}$, $m{{=}}{m:.2f}$', fontsize=7.5, color=col, pad=1.5)
    if is_switch:
        ax.text(0.05, 0.95, r'$\Rightarrow f_{%d}$' % (cur+1), transform=ax.transAxes,
                ha='left', va='top', fontsize=7.5, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.10', fc='#1565c0', ec='none', alpha=0.9))
    delta = np.clip(0.5 + 8.0*(xadv-x0), 0, 1)
    iax = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
    iax.imshow(np.transpose(delta, (1, 2, 0)), interpolation='nearest')
    iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values(): sp.set_color('k'); sp.set_linewidth(0.5)


fig, axes = plt.subplots(2, 3, figsize=(3.4, 2.55))
for c, q in enumerate(qs):
    xa, cu = s_snaps[q]; cell(axes[0, c], xa, cu, q)
    xa, cu = m_snaps[q]; cell(axes[1, c], xa, cu, q, is_switch=(q == m_sw[0]))
axes[0, 0].set_ylabel('Static', fontsize=9, fontweight='bold', labelpad=2)
axes[1, 0].set_ylabel('MADS', fontsize=9, fontweight='bold', labelpad=2)
plt.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.01, wspace=0.05, hspace=0.24)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(FIG, f'fig_attack_progression.{ext}'), dpi=200, bbox_inches='tight')
print('saved fig_attack_progression', flush=True)
