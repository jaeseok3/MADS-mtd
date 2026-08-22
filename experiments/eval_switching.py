"""
Evaluate MTD switching strategies against score-based black-box attacks.

Strategies:
  - static: No switching (single model)
  - periodic-N: Switch every N queries
  - random-p: Switch with probability p per query
  - signal: Switch when running margin drops below threshold

Usage:
    python eval_switching.py --pool_dir divergent_kd_v5b_pool --num_images 200 --device cuda
"""
import sys, os, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.resnet18_cifar100 import ResNet18_CIFAR100

DATA_ROOT = "data"
CHECKPOINT_DIR = "checkpoints"
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
EPSILON = 8.0 / 255.0


class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def load_pool(pool_dir, device):
    ckpt_dir = os.path.join(CHECKPOINT_DIR, pool_dir)
    models = []
    for fname in sorted(os.listdir(ckpt_dir)):
        if fname.endswith('.pth'):
            m = ResNet18_CIFAR100(num_classes=100)
            m.load_state_dict(torch.load(os.path.join(ckpt_dir, fname),
                                         map_location="cpu", weights_only=True))
            wrapped = NormalizedModel(m, CIFAR100_MEAN, CIFAR100_STD)
            wrapped.to(device).eval()
            for p in wrapped.parameters():
                p.requires_grad = False
            models.append(wrapped)
    return models


class SwitchingOracle:
    def __init__(self, models, device, strategy='static', **kwargs):
        self.models = models
        self.device = device
        self.K = len(models)
        self.strategy = strategy
        self.kwargs = kwargs

        self.current_idx = 0
        self.query_count = 0
        self.switches = 0
        self.margin_history = deque(maxlen=kwargs.get('window', 20))

        self.switch_interval = kwargs.get('interval', 100)
        self.switch_prob = kwargs.get('prob', 0.01)
        self.margin_threshold = kwargs.get('margin_thresh', 0.5)
        self.queries_since_switch = 0

    def reset(self, start_idx=0):
        self.current_idx = start_idx
        self.query_count = 0
        self.switches = 0
        self.margin_history.clear()
        self.queries_since_switch = 0

    def _switch(self):
        old = self.current_idx
        candidates = [i for i in range(self.K) if i != self.current_idx]
        self.current_idx = np.random.choice(candidates)
        self.switches += 1
        self.queries_since_switch = 0
        self.margin_history.clear()

    def _should_switch(self, scores):
        if self.strategy == 'static':
            return False

        if self.strategy.startswith('periodic'):
            return self.queries_since_switch >= self.switch_interval

        if self.strategy.startswith('random'):
            return np.random.random() < self.switch_prob

        if self.strategy == 'signal':
            pred = np.argmax(scores)
            sorted_s = np.sort(scores)[::-1]
            margin = sorted_s[0] - sorted_s[1]
            self.margin_history.append(margin)
            if len(self.margin_history) >= 10:
                recent = list(self.margin_history)[-10:]
                avg_margin = np.mean(recent)
                if avg_margin < self.margin_threshold:
                    return True
            return False

        return False

    def __call__(self, x_batch):
        if isinstance(x_batch, np.ndarray):
            x_batch = torch.from_numpy(x_batch).float()
        x_batch = x_batch.to(self.device)
        self.query_count += x_batch.shape[0]
        self.queries_since_switch += x_batch.shape[0]

        with torch.no_grad():
            logits = self.models[self.current_idx](x_batch)
            scores = F.softmax(logits, dim=1).cpu().numpy()

        if x_batch.shape[0] == 1:
            if self._should_switch(scores[0]):
                self._switch()

        return scores


def clip_eps(x_adv, x_orig, eps):
    delta = np.clip(x_adv - x_orig, -eps, eps)
    return np.clip(x_orig + delta, 0.0, 1.0)


def square_attack(oracle, x_orig, true_label, max_queries, eps=EPSILON, p_init=0.8):
    C, H, W = x_orig.shape
    x_adv = x_orig.copy()
    init_delta = np.random.choice([-eps, eps], size=(C, H, W)).astype(np.float32)
    x_adv = np.clip(x_orig + init_delta, 0.0, 1.0)

    scores = oracle(x_adv[None])
    if np.argmax(scores) != true_label:
        return x_adv, oracle.query_count, True
    margin_best = scores[0, true_label] - np.max(np.delete(scores[0], true_label))

    for i in range(1, max_queries):
        if oracle.query_count >= max_queries:
            break
        p = max(p_init * (1.0 - i / max_queries), 0.01)
        s = max(int(np.ceil(np.sqrt(p * H * W))), 1)
        s = min(s, H)
        y0 = np.random.randint(0, H - s + 1)
        x0 = np.random.randint(0, W - s + 1)
        x_cand = x_adv.copy()
        patch = np.random.choice([-2 * eps, 2 * eps], size=(C, s, s)).astype(np.float32)
        x_cand[:, y0:y0+s, x0:x0+s] = x_cand[:, y0:y0+s, x0:x0+s] + patch
        x_cand = clip_eps(x_cand, x_orig, eps)
        scores = oracle(x_cand[None])
        if np.argmax(scores) != true_label:
            return x_cand, oracle.query_count, True
        margin = scores[0, true_label] - np.max(np.delete(scores[0], true_label))
        if margin < margin_best:
            x_adv = x_cand
            margin_best = margin

    return x_adv, oracle.query_count, (np.argmax(oracle(x_adv[None])) != true_label)


def simba_dct_attack(oracle, x_orig, true_label, max_queries, eps=EPSILON, freq_limit=16):
    C, H, W = x_orig.shape
    bases = []
    for i in range(freq_limit):
        for j in range(freq_limit):
            if i == 0 and j == 0:
                continue
            v = np.zeros((H, H), dtype=np.float32)
            xs = np.arange(H)
            for a in range(H):
                v[a, :] = np.cos(np.pi * (2*a+1)*i / (2*H)) * \
                           np.cos(np.pi * (2*xs+1)*j / (2*H))
            v /= np.linalg.norm(v)
            for c in range(C):
                full = np.zeros((C, H, H), dtype=np.float32)
                full[c] = v
                bases.append(full)

    perm = np.random.permutation(len(bases))
    x_adv = x_orig.copy()
    scores = oracle(x_adv[None])
    if np.argmax(scores) != true_label:
        return x_adv, oracle.query_count, True
    best_score = scores[0, true_label]

    for idx in range(len(perm)):
        if oracle.query_count >= max_queries:
            break
        q = bases[perm[idx]]
        for sign in [1, -1]:
            if oracle.query_count >= max_queries:
                break
            x_new = clip_eps(x_adv + sign * eps * q, x_orig, eps)
            scores = oracle(x_new[None])
            if np.argmax(scores) != true_label:
                return x_new, oracle.query_count, True
            if scores[0, true_label] < best_score:
                x_adv = x_new
                best_score = scores[0, true_label]
                break

    return x_adv, oracle.query_count, (np.argmax(oracle(x_adv[None])) != true_label)


def signhunter_attack(oracle, x_orig, true_label, max_queries, eps=EPSILON):
    C, H, W = x_orig.shape
    dim = C * H * W
    sign = np.ones(dim, dtype=np.float32)
    x_adv = clip_eps(x_orig + eps * sign.reshape(C, H, W), x_orig, eps)
    scores = oracle(x_adv[None])
    if np.argmax(scores) != true_label:
        return x_adv, oracle.query_count, True
    best_score = scores[0, true_label]

    perm = np.random.permutation(dim)
    chunk_size = dim
    while oracle.query_count < max_queries and chunk_size >= 1:
        for start in range(0, dim, chunk_size):
            if oracle.query_count >= max_queries:
                break
            end = min(start + chunk_size, dim)
            indices = perm[start:end]
            sign_new = sign.copy()
            sign_new[indices] *= -1
            x_new = clip_eps(x_orig + eps * sign_new.reshape(C, H, W), x_orig, eps)
            scores = oracle(x_new[None])
            if np.argmax(scores) != true_label:
                return x_new, oracle.query_count, True
            if scores[0, true_label] < best_score:
                sign = sign_new
                best_score = scores[0, true_label]
                x_adv = x_new
        chunk_size //= 2

    return x_adv, oracle.query_count, (np.argmax(oracle(x_adv[None])) != true_label)


ATTACKS = {
    "SquareAttack": square_attack,
    "SimBA-DCT": simba_dct_attack,
    "SignHunter": signhunter_attack,
}


def run_evaluation(pool, device, testset, correct_indices, strategy, strategy_kwargs,
                   attacks, max_queries=1000, num_images=200, seed=42):
    np.random.seed(seed)
    selected = np.random.choice(correct_indices, min(num_images, len(correct_indices)),
                                replace=False)
    results = {}
    for atk_name, atk_fn in attacks.items():
        successes = 0
        total_queries = 0
        total_switches = 0
        queries_to_success = []

        oracle = SwitchingOracle(pool, device, strategy=strategy, **strategy_kwargs)

        for img_idx in selected:
            x, y = testset[img_idx]
            if isinstance(x, torch.Tensor):
                x = x.numpy()

            oracle.reset(start_idx=np.random.randint(0, len(pool)))
            np.random.seed(seed + img_idx)
            _, queries, success = atk_fn(oracle, x, y, max_queries)
            if success:
                successes += 1
                queries_to_success.append(queries)
            total_queries += queries
            total_switches += oracle.switches

        asr = successes / len(selected) * 100
        avg_q = np.mean(queries_to_success) if queries_to_success else max_queries
        avg_switches = total_switches / len(selected)
        results[atk_name] = {
            'asr': float(asr),
            'avg_queries_success': float(avg_q),
            'avg_switches': float(avg_switches),
            'num_images': len(selected),
        }
        print(f"    {atk_name}: ASR={asr:.1f}% | AvgQ={avg_q:.0f} | Switches={avg_switches:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--max_queries", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tag", type=str, default="switching")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    print(f"[*] Loading pool from: {args.pool_dir}")

    pool = load_pool(args.pool_dir, device)
    print(f"[*] Pool size: {len(pool)} models")

    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR100(root=DATA_ROOT, train=False, download=False, transform=transform)

    print("[*] Selecting images correctly classified by ALL models...")
    correct_mask = None
    loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    with torch.no_grad():
        all_correct = []
        idx = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            batch_correct = torch.ones(labels.size(0), dtype=torch.bool, device=device)
            for m in pool:
                batch_correct &= (m(images).argmax(1) == labels)
            all_correct.append(batch_correct.cpu())
    correct_mask = torch.cat(all_correct)
    correct_indices = correct_mask.nonzero(as_tuple=False).squeeze().numpy()
    print(f"[*] Correctly classified by all: {len(correct_indices)}")

    strategies = [
        ('static', {}),
        ('periodic', {'interval': 50}),
        ('periodic', {'interval': 100}),
        ('periodic', {'interval': 200}),
        ('random', {'prob': 0.02}),
        ('random', {'prob': 0.01}),
        ('random', {'prob': 0.005}),
        ('signal', {'margin_thresh': 0.3, 'window': 20}),
        ('signal', {'margin_thresh': 0.5, 'window': 20}),
        ('signal', {'margin_thresh': 0.7, 'window': 20}),
    ]

    all_results = {}
    t0 = time.time()

    for strategy, kwargs in strategies:
        if kwargs:
            key_str = '_'.join(f"{k}{v}" for k, v in kwargs.items())
            label = f"{strategy}_{key_str}"
        else:
            label = strategy
        print(f"\n[*] Strategy: {label}")
        results = run_evaluation(
            pool, device, testset, correct_indices,
            strategy=strategy, strategy_kwargs=kwargs,
            attacks=ATTACKS, max_queries=args.max_queries,
            num_images=args.num_images, seed=42)
        all_results[label] = results

    save_path = os.path.join(OUT_DIR, f"switching_results_{args.tag}.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[*] Saved to {save_path}")
    print(f"[*] Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
