"""Train MTD student pools on TinyImageNet-200 for the Challenge-1 comparison.
Methods (one per --method):
  div_kd : all-pairs repulsive Divergent KD (warmup standard KD, then repulsion)
  std_kd : standard KD (beta=0)
  independent : 5 nets trained from scratch on full data (no teacher)
  eimtd : 5 nets each on a disjoint 1/K data shard (no teacher)
  deepmtd : 1 base (= a std_kd student) + Gaussian-noised copies (no training)
Uses cached in-RAM data (tin_cache.py) + AMP. Saves checkpoints/<tag>/student_k.pth."""
import os, sys, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tin_common as T

EPS = T.EPSILON; PGD_STEPS = 20; PGD_ALPHA = 2.0 / 255.0
torch.backends.cudnn.benchmark = True


def pgd(model, x, y, eps=EPS, alpha=PGD_ALPHA, steps=PGD_STEPS):
    adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    adv = adv.clamp(0, 1)
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        g = torch.autograd.grad(loss, adv)[0]
        adv = (adv.detach() + alpha * g.sign())
        adv = torch.min(torch.max(adv, x - eps), x + eps).clamp(0, 1).detach()
    return adv


@torch.no_grad()
def eval_pool(students, xva, yva, device):
    for s in students: s.eval()
    K = len(students); N = xva.shape[0]
    preds = [np.zeros(N, dtype=np.int64) for _ in range(K)]
    labs = yva.numpy()
    for s0 in range(0, N, 512):
        e = min(s0 + 512, N)
        xb = xva[s0:e].to(device).float().div_(255.0)
        for k, s in enumerate(students):
            preds[k][s0:e] = s(xb).argmax(1).cpu().numpy()
    accs = [float((preds[k] == labs).mean() * 100) for k in range(K)]
    agr = [float((preds[i] == preds[j]).mean() * 100) for i in range(K) for j in range(i+1, K)]
    return {"accs": accs, "acc_mean": float(np.mean(accs)),
            "agreement": float(np.mean(agr)) if agr else 0.0}


def measure_transfer(students, xva, yva, device, num_images=500):
    for s in students: s.eval()
    K = len(students); N = xva.shape[0]
    # indices correctly classified by all students
    bc = np.ones(N, dtype=bool); labs = yva.numpy()
    with torch.no_grad():
        for s0 in range(0, N, 512):
            e = min(s0 + 512, N); xb = xva[s0:e].to(device).float().div_(255.0)
            agree = np.ones(e - s0, dtype=bool)
            for s in students:
                agree &= (s(xb).argmax(1).cpu().numpy() == labs[s0:e])
            bc[s0:e] = agree
    idx = np.nonzero(bc)[0]
    if len(idx) == 0:
        return {"transfer_matrix": [], "self_asr": 0.0, "cross_asr": 0.0, "num_images": 0}
    rng = np.random.RandomState(42)
    if len(idx) > num_images: idx = rng.choice(idx, num_images, replace=False)
    X = xva[idx].to(device).float().div_(255.0); Y = torch.from_numpy(labs[idx]).to(device)
    M = np.zeros((K, K)); n = X.shape[0]
    for src in range(K):
        advs = []
        for s0 in range(0, n, 64):
            e = min(s0 + 64, n)
            advs.append(pgd(students[src], X[s0:e], Y[s0:e]))
        A = torch.cat(advs)
        for tgt in range(K):
            with torch.no_grad():
                mis = 0
                for s0 in range(0, n, 64):
                    e = min(s0 + 64, n)
                    mis += (students[tgt](A[s0:e]).argmax(1) != Y[s0:e]).sum().item()
                M[src][tgt] = mis / n * 100
    self_r = [M[i][i] for i in range(K)]
    cross = [M[i][j] for i in range(K) for j in range(K) if i != j]
    return {"transfer_matrix": M.tolist(), "self_asr": float(np.mean(self_r)),
            "cross_asr": float(np.mean(cross)), "num_images": int(len(idx))}


def new_student(device, seed, arch="resnet"):
    torch.manual_seed(seed)
    return T.wrap(T.make_model(arch=arch), device, eval_mode=False)


def student_from_teacher(teacher, device, seed, arch="resnet", noise=0.02):
    """Warm-start a student by copying teacher weights + per-student Gaussian
    noise (relative to each tensor's std) to break symmetry, so all-pairs
    repulsion can diverge them from the shared basin."""
    torch.manual_seed(seed)
    s = T.wrap(T.make_model(arch=arch), device, eval_mode=False)
    tsd = teacher.model.state_dict()
    sd = {}
    for k, v in tsd.items():
        v = v.clone()
        if v.dtype.is_floating_point and v.numel() > 1:
            v = v + noise * v.float().std() * torch.randn_like(v.float()).to(v.dtype)
        sd[k] = v
    s.model.load_state_dict(sd)
    return s


def train_kd(args, device, method, xtr, ytr, xva, yva):
    """div_kd or std_kd (std_kd = beta 0). Returns list of students."""
    teacher = T.load_teacher(device, arch=args.arch)
    K = args.num_students; beta = 0.0 if method == "std_kd" else args.beta
    students, opts, scheds = [], [], []
    for i in range(K):
        if getattr(args, "teacher_init", False):
            s = student_from_teacher(teacher, device, 300 + i, args.arch, args.init_noise); s.train()
        else:
            s = new_student(device, 300 + i, args.arch); s.train()
        _lr = args.lr
        if getattr(args, "resume_tag", ""):
            _ck = os.path.join(T.CHECKPOINT_DIR, args.resume_tag, f"student_{i}.pth")
            s.model.load_state_dict(torch.load(_ck, map_location=device, weights_only=True)); s.train()
            _lr = args.resume_lr
        o = torch.optim.SGD(s.parameters(), lr=_lr, momentum=0.9, weight_decay=5e-4)
        students.append(s); opts.append(o)
        scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs))
    scaler = torch.cuda.amp.GradScaler()
    tl = T.GPULoader(xtr, ytr, device, batch_size=args.bs, shuffle=True, augment=True, drop_last=True)
    t0 = time.time()
    for ep in range(args.epochs):
        warm = ep < args.warmup_epochs
        for x, y in tl:
            with torch.no_grad(), torch.cuda.amp.autocast():
                tlog = teacher(x)
            if warm or method == "std_kd":
                for k in range(K):
                    opts[k].zero_grad()
                    with torch.cuda.amp.autocast():
                        lg = students[k](x)
                        kl = F.kl_div(F.log_softmax(lg/args.T, 1), F.softmax(tlog/args.T, 1),
                                      reduction='batchmean') * args.T * args.T
                        loss = args.alpha * F.cross_entropy(lg, y) + args.lambda_ * kl
                    scaler.scale(loss).backward()
                    scaler.unscale_(opts[k])
                    torch.nn.utils.clip_grad_norm_(students[k].parameters(), 5.0)
                    scaler.step(opts[k])
                scaler.update()  # once per batch (standard multi-optimizer pattern)
            else:
                # precompute detached softmaxes of all students (no graph held)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    sm = [F.softmax(students[k](x)/args.T, 1) for k in range(K)]
                    tsm = F.softmax(tlog/args.T, 1)
                # then update each student sequentially with only its own graph alive
                for k in range(K):
                    opts[k].zero_grad()
                    with torch.cuda.amp.autocast():
                        lg = students[k](x)
                        ls = F.log_softmax(lg/args.T, 1)
                        klt = F.kl_div(ls, tsm, reduction='batchmean') * args.T * args.T
                        rep = torch.zeros((), device=device)
                        for j in range(K):
                            if j == k: continue
                            klkj = F.kl_div(ls, sm[j], reduction='none').sum(1) * args.T * args.T
                            rep = rep + F.relu(args.kl_target - klkj).mean()
                        rep = rep / (K - 1)
                        loss = args.alpha * F.cross_entropy(lg, y) + args.lambda_ * klt + beta * rep
                    scaler.scale(loss).backward()
                    scaler.unscale_(opts[k])
                    torch.nn.utils.clip_grad_norm_(students[k].parameters(), 5.0)
                    scaler.step(opts[k])
                scaler.update()  # once per batch
        for k in range(K): scheds[k].step()
        m = eval_pool(students, xva, yva, device)
        print(f"[{method}][Ep {ep+1:2d}/{args.epochs}] {'WARM' if warm else 'DIV'} "
              f"acc={m['acc_mean']:.2f}% agree={m['agreement']:.1f}% {time.time()-t0:.0f}s", flush=True)
    return students


def train_independent(args, device, xtr, ytr, xva, yva, shards=False):
    K = args.num_students; N = xtr.shape[0]
    perm = np.random.RandomState(0).permutation(N)
    students = []
    for i in range(K):
        s = new_student(device, 500 + i, args.arch); s.train()
        _lr = args.lr
        if getattr(args, "resume_tag", ""):
            _ck = os.path.join(T.CHECKPOINT_DIR, args.resume_tag, f"student_{i}.pth")
            s.model.load_state_dict(torch.load(_ck, map_location=device, weights_only=True)); s.train()
            _lr = args.resume_lr
        o = torch.optim.SGD(s.parameters(), lr=_lr, momentum=0.9, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs)
        scaler = torch.cuda.amp.GradScaler()
        if shards:
            sl = perm[i*(N//K):(i+1)*(N//K)]
            xi, yi = xtr[sl], ytr[sl]
        else:
            xi, yi = xtr, ytr
        tl = T.GPULoader(xi, yi, device, batch_size=args.bs, shuffle=True, augment=True, drop_last=True)
        t0 = time.time()
        for ep in range(args.epochs):
            for x, y in tl:
                o.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = F.cross_entropy(s(x), y)
                scaler.scale(loss).backward()
                scaler.unscale_(o)
                torch.nn.utils.clip_grad_norm_(s.parameters(), 5.0)
                scaler.step(o); scaler.update()
            sch.step()
        s.eval()
        print(f"[{'eimtd' if shards else 'independent'}] student {i} done {time.time()-t0:.0f}s", flush=True)
        students.append(s)
    return students


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["div_kd", "std_kd", "independent", "eimtd", "deepmtd"])
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lambda_", type=float, default=0.3)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--kl_target", type=float, default=5.0)
    ap.add_argument("--warmup_epochs", type=int, default=4)
    ap.add_argument("--num_students", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arch", type=str, default="resnet")
    ap.add_argument("--teacher_init", action="store_true",
                    help="warm-start students from teacher weights (+noise) instead of random init")
    ap.add_argument("--resume_tag", type=str, default="",
                    help="warm-start each student from checkpoints/<resume_tag>/student_k.pth and train only --epochs MORE")
    ap.add_argument("--resume_lr", type=float, default=0.01,
                    help="(fine-tune) lr used when --resume_tag is set")
    ap.add_argument("--init_noise", type=float, default=0.02,
                    help="per-tensor relative Gaussian noise for teacher-init symmetry breaking")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] method={args.method} tag={args.tag} beta={args.beta} klT={args.kl_target} "
          f"ep={args.epochs} bs={args.bs}", flush=True)
    xtr, ytr = T.load_cache("train"); xva, yva = T.load_cache("val")

    if args.method in ("div_kd", "std_kd"):
        students = train_kd(args, device, args.method, xtr, ytr, xva, yva)
    elif args.method == "independent":
        students = train_independent(args, device, xtr, ytr, xva, yva, shards=False)
    elif args.method == "eimtd":
        students = train_independent(args, device, xtr, ytr, xva, yva, shards=True)
    elif args.method == "deepmtd":
        # DeepMTD = one base model + Gaussian-noised copies. Reuse a trained
        # std_kd student as the base if available; else train a 1-student base.
        _base_tag = args.resume_tag if getattr(args, "resume_tag", "") else (f"tin_std_kd_{args.arch}" if args.arch!="resnet" else "tin_std_kd")
        base_ck = os.path.join(T.CHECKPOINT_DIR, _base_tag, "student_0.pth")
        if os.path.exists(base_ck):
            bm = T.make_model(arch=args.arch); bm.load_state_dict(torch.load(base_ck, map_location="cpu", weights_only=True))
            base = T.wrap(bm, device)
            print("[deepmtd] reusing tin_std_kd/student_0 as base", flush=True)
        else:
            a2 = argparse.Namespace(**{**vars(args), "num_students": 1})
            base = train_kd(a2, device, "std_kd", xtr, ytr, xva, yva)[0]
        students = [base]; bsd = base.model.state_dict()
        for i in range(1, args.num_students):
            m = T.make_model(arch=args.arch)
            sd = {k: v.clone() for k, v in bsd.items()}
            for k in sd:
                if sd[k].dtype.is_floating_point:
                    sd[k] = sd[k] + 0.01 * torch.randn_like(sd[k])
            m.load_state_dict(sd); students.append(T.wrap(m, device))

    metrics = eval_pool(students, xva, yva, device)
    transfer = measure_transfer(students, xva, yva, device)
    print(f"[*] {args.tag}: acc={metrics['acc_mean']:.2f}% agree={metrics['agreement']:.1f}% "
          f"self_asr={transfer['self_asr']:.1f}% cross_asr={transfer['cross_asr']:.1f}%", flush=True)

    ckpt_dir = os.path.join(T.CHECKPOINT_DIR, args.tag); os.makedirs(ckpt_dir, exist_ok=True)
    for k, s in enumerate(students):
        torch.save(s.model.state_dict(), os.path.join(ckpt_dir, f"student_{k}.pth"))
    out = {"method": args.method, "tag": args.tag, "metrics": metrics, "transfer": transfer}
    json.dump(out, open(f"tin_pool_{args.tag}.json", "w"), indent=2)
    print(f"[done] saved {len(students)} students to {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
