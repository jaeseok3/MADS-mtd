"""CIFAR-10 pool training - mirrors tin_train_pools.py exactly (same trainers,
same hyperparameter conventions), swapping the dataset for CIFAR-10 (10 classes,
32x32, pad-4 crop aug). Methods:
  teacher     : 200-epoch SGD teacher (same recipe as train_all_teachers.py)
  independent : K students, CE only, distinct seeds
  std_kd      : K students, alpha*CE + lambda*KL(teacher) (beta=0)
  div_kd      : + beta * all-pairs hinge KL repulsion after warmup
Checkpoints -> checkpoints/<tag>/student_k.pth (teacher -> checkpoints/<arch>_c10_teacher.pth)
"""
import os, sys, json, time, argparse
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_common as A

DATA_ROOT = A.DATA_ROOT
CKPT = A.CHECKPOINT_DIR
C10_NORM = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
NUM_CLASSES = 10


def wrap(model, device, eval_mode=True):
    w = A.NormalizedModel(model, *C10_NORM); w.to(device)
    if eval_mode:
        w.eval()
        for p in w.parameters():
            p.requires_grad = False
    return w


def load_c10(split):
    """CIFAR-10 as in-memory uint8 NCHW tensors (downloads on first use)."""
    from torchvision import datasets
    ds = datasets.CIFAR10(root=DATA_ROOT, train=(split == 'train'), download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x, y


def _gpu_random_crop(x, pad=4):
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    ty = torch.randint(0, 2*pad+1, (B,), device=x.device)
    tx = torch.randint(0, 2*pad+1, (B,), device=x.device)
    ar = torch.arange(H, device=x.device)
    h_idx = (ty[:, None] + ar[None, :])
    w_idx = (tx[:, None] + ar[None, :])
    out = xp[torch.arange(B)[:, None, None, None], torch.arange(C)[None, :, None, None],
             h_idx[:, None, :, None], w_idx[:, None, None, :]]
    return out


class GPULoader:
    def __init__(self, x_u8, y, device, batch_size=128, shuffle=True, augment=False, drop_last=False):
        self.x = x_u8; self.y = y; self.device = device
        self.bs = batch_size; self.shuffle = shuffle; self.augment = augment
        self.drop_last = drop_last; self.N = x_u8.shape[0]

    def __iter__(self):
        order = torch.randperm(self.N) if self.shuffle else torch.arange(self.N)
        last = self.N - (self.N % self.bs) if self.drop_last else self.N
        for s in range(0, last, self.bs):
            idx = order[s:s+self.bs]
            xb = self.x[idx].to(self.device, non_blocking=True).float().div_(255.0)
            yb = self.y[idx].to(self.device, non_blocking=True)
            if self.augment:
                xb = _gpu_random_crop(xb, pad=4)
                flip = torch.rand(xb.shape[0], device=self.device) < 0.5
                xb[flip] = torch.flip(xb[flip], dims=[3])
            yield xb, yb


def eval_pool(students, xva, yva, device):
    K = len(students); labs = yva.numpy(); N = xva.shape[0]
    preds = np.zeros((K, N), dtype=np.int64)
    with torch.no_grad():
        for s0 in range(0, N, 512):
            e = min(s0+512, N); xb = xva[s0:e].to(device).float().div_(255.0)
            for k in range(K):
                preds[k, s0:e] = students[k](xb).argmax(1).cpu().numpy()
    accs = [float((preds[k] == labs).mean()*100) for k in range(K)]
    agr = [float((preds[i] == preds[j]).mean()*100) for i in range(K) for j in range(i+1, K)]
    return {"accs": accs, "acc_mean": float(np.mean(accs)),
            "agreement": float(np.mean(agr)) if agr else 0.0}


def measure_transfer(students, xva, yva, device, eps=8/255, steps=20, cap=300):
    K = len(students); labs = yva.numpy(); N = xva.shape[0]
    bc = np.ones(N, bool)
    with torch.no_grad():
        for s0 in range(0, N, 512):
            e = min(s0+512, N); xb = xva[s0:e].to(device).float().div_(255.0)
            ag = np.ones(e-s0, bool)
            for m in students: ag &= (m(xb).argmax(1).cpu().numpy() == labs[s0:e])
            bc[s0:e] = ag
    idx = np.nonzero(bc)[0]; rng = np.random.RandomState(0)
    if len(idx) > cap: idx = rng.choice(idx, cap, replace=False)
    X = xva[idx].to(device).float().div_(255.0); Y = torch.from_numpy(labs[idx]).to(device); n = X.shape[0]
    alpha = max(eps/4, 1/255); M = np.zeros((K, K))
    for src in range(K):
        for p in students[src].parameters(): p.requires_grad = False
        adv = (X.clone() + torch.empty_like(X).uniform_(-eps, eps)).clamp(0, 1)
        for _ in range(steps):
            adv.requires_grad_(True)
            loss = F.cross_entropy(students[src](adv), Y)
            g = torch.autograd.grad(loss, adv)[0]
            adv = torch.min(torch.max(adv.detach()+alpha*g.sign(), X-eps), X+eps).clamp(0, 1).detach()
        with torch.no_grad():
            for tgt in range(K):
                M[src][tgt] = float((students[tgt](adv).argmax(1) != Y).float().mean()*100)
    self_r = [M[i][i] for i in range(K)]
    cross = [M[i][j] for i in range(K) for j in range(K) if i != j]
    return {"self_asr": float(np.mean(self_r)), "cross_asr": float(np.mean(cross)), "num_images": n}


def new_student(device, seed, arch):
    torch.manual_seed(seed)
    return wrap(A.make_model(arch, NUM_CLASSES), device, eval_mode=False)


def load_teacher(device, arch):
    m = A.make_model(arch, NUM_CLASSES)
    m.load_state_dict(torch.load(os.path.join(CKPT, f"{arch}_c10_teacher.pth"),
                                 map_location="cpu", weights_only=True))
    return wrap(m, device)


def train_teacher(args, device, xtr, ytr, xva, yva):
    s = new_student(device, 100, args.arch); s.train()
    o = torch.optim.SGD(s.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    tl = GPULoader(xtr, ytr, device, batch_size=args.bs, shuffle=True, augment=True, drop_last=True)
    best = 0.0; t0 = time.time()
    for ep in range(args.epochs):
        s.train()
        for x, y in tl:
            o.zero_grad()
            with torch.cuda.amp.autocast():
                loss = F.cross_entropy(s(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(o)
            torch.nn.utils.clip_grad_norm_(s.parameters(), 5.0)
            scaler.step(o); scaler.update()
        sch.step()
        s.eval(); acc = eval_pool([s], xva, yva, device)["acc_mean"]
        if acc > best:
            best = acc
            torch.save(s.model.state_dict(), os.path.join(CKPT, f"{args.arch}_c10_teacher.pth"))
        if (ep+1) % 10 == 0:
            print(f"[teacher][Ep {ep+1:3d}/{args.epochs}] acc={acc:.2f}% best={best:.2f}% {time.time()-t0:.0f}s", flush=True)
    print(f"[*] {args.arch}_c10_teacher: best acc={best:.2f}%", flush=True)


def train_kd(args, device, method, xtr, ytr, xva, yva):
    teacher = load_teacher(device, args.arch)
    K = args.num_students; beta = 0.0 if method == "std_kd" else args.beta
    students, opts, scheds = [], [], []
    for i in range(K):
        s = new_student(device, 300+i, args.arch); s.train()
        o = torch.optim.SGD(s.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        students.append(s); opts.append(o)
        scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs))
    scaler = torch.cuda.amp.GradScaler()
    tl = GPULoader(xtr, ytr, device, batch_size=args.bs, shuffle=True, augment=True, drop_last=True)
    t0 = time.time()
    for ep in range(args.epochs):
        warm = ep < args.warmup_epochs
        for s in students: s.train()
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
                scaler.update()
            else:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    sm = [F.softmax(students[k](x)/args.T, 1) for k in range(K)]
                    tsm = F.softmax(tlog/args.T, 1)
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
                        rep = rep / (K-1)
                        loss = args.alpha * F.cross_entropy(lg, y) + args.lambda_ * klt + beta * rep
                    scaler.scale(loss).backward()
                    scaler.unscale_(opts[k])
                    torch.nn.utils.clip_grad_norm_(students[k].parameters(), 5.0)
                    scaler.step(opts[k])
                scaler.update()
        for k in range(K): scheds[k].step()
        for s in students: s.eval()
        m = eval_pool(students, xva, yva, device)
        print(f"[{method}][Ep {ep+1:2d}/{args.epochs}] {'WARM' if warm else 'DIV'} "
              f"acc={m['acc_mean']:.2f}% agree={m['agreement']:.1f}% {time.time()-t0:.0f}s", flush=True)
    return students


def train_independent(args, device, xtr, ytr, xva, yva):
    K = args.num_students; students = []
    for i in range(K):
        s = new_student(device, 500+i, args.arch); s.train()
        o = torch.optim.SGD(s.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs)
        scaler = torch.cuda.amp.GradScaler()
        tl = GPULoader(xtr, ytr, device, batch_size=args.bs, shuffle=True, augment=True, drop_last=True)
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
        print(f"[independent] student {i} done {time.time()-t0:.0f}s", flush=True)
        students.append(s)
    return students


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet", choices=["resnet", "vgg", "alexnet"])
    ap.add_argument("--method", required=True,
                    choices=["teacher", "independent", "std_kd", "div_kd"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--num_students", type=int, default=5)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--lambda_", type=float, default=0.3)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--kl_target", type=float, default=5.0)
    args = ap.parse_args()

    device = torch.device("cuda")
    xtr, ytr = load_c10("train"); xva, yva = load_c10("val")
    print(f"[data] CIFAR-10 train={xtr.shape} val={xva.shape}", flush=True)

    if args.method == "teacher":
        args_epochs = args.epochs if args.epochs != 20 else 200  # teacher default 200
        args.epochs = args_epochs
        train_teacher(args, device, xtr, ytr, xva, yva)
        return

    if args.method == "independent":
        students = train_independent(args, device, xtr, ytr, xva, yva)
    else:
        students = train_kd(args, device, args.method, xtr, ytr, xva, yva)

    metrics = eval_pool(students, xva, yva, device)
    transfer = measure_transfer(students, xva, yva, device)
    print(f"[*] {args.tag}: acc={metrics['acc_mean']:.2f}% agree={metrics['agreement']:.1f}% "
          f"self_asr={transfer['self_asr']:.1f}% cross_asr={transfer['cross_asr']:.1f}%", flush=True)

    d = os.path.join(CKPT, args.tag); os.makedirs(d, exist_ok=True)
    for k, s in enumerate(students):
        torch.save(s.model.state_dict(), os.path.join(d, f"student_{k}.pth"))
    json.dump({"metrics": metrics, "transfer": transfer, "args": vars(args)},
              open(os.path.join(d, "summary.json"), "w"), indent=2)
    print(f"[saved] {d}", flush=True)


if __name__ == "__main__":
    main()
