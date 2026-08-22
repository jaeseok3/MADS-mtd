"""
Train students via All-Pairs Repulsive Divergent KD on CIFAR-100.

Two-phase training:
  Phase 1 (warmup): Standard KD
  Phase 2 (diversify): Each student is pushed away from ALL other students

Loss (Phase 2):
  L = alpha * CE(s_k, y) + lambda * KL(s_k || teacher)
    + beta * (1/(K-1)) * sum_{j!=k} max(0, kl_target - KL(s_k || s_j))

The repulsion term pushes each student to be at least kl_target KL-divergent
from every other student, creating maximum pairwise diversity.

Usage:
    python train_divergent_kd_v5_allpairs.py [--warmup_epochs 5] [--kl_target 5] [--tag v5b_default]
"""
import sys, os, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.resnet18_cifar100 import ResNet18_CIFAR100

DATA_ROOT = "data"
CHECKPOINT_DIR = "checkpoints"
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

EPSILON = 8.0 / 255.0
PGD_STEPS = 20
PGD_ALPHA = 2.0 / 255.0


class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def load_teacher(device):
    model = ResNet18_CIFAR100(num_classes=100)
    ckpt = os.path.join(CHECKPOINT_DIR, "resnet18_cifar100_teacher.pth")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    wrapped = NormalizedModel(model, CIFAR100_MEAN, CIFAR100_STD)
    wrapped.to(device).eval()
    for p in wrapped.parameters():
        p.requires_grad = False
    return wrapped


def create_student(device, seed):
    torch.manual_seed(seed)
    model = ResNet18_CIFAR100(num_classes=100)
    wrapped = NormalizedModel(model, CIFAR100_MEAN, CIFAR100_STD)
    wrapped.to(device)
    return wrapped


def pgd_attack(model, images, labels, eps, alpha, steps):
    adv = images.clone().detach()
    adv += torch.empty_like(adv).uniform_(-eps, eps)
    adv = torch.clamp(adv, 0, 1)
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), labels)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        delta = torch.clamp(adv - images, -eps, eps)
        adv = torch.clamp(images + delta, 0, 1).detach()
    return adv


@torch.no_grad()
def evaluate_students(students, testloader, device):
    K = len(students)
    all_preds, all_probs, all_labels = [], [], []
    for s in students:
        s.eval()
    for images, labels in testloader:
        images = images.to(device)
        all_labels.append(labels.numpy())
        batch_preds, batch_probs = [], []
        for s in students:
            logits = s(images)
            batch_preds.append(logits.argmax(dim=1).cpu().numpy())
            batch_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_preds.append(np.stack(batch_preds, axis=0))
        all_probs.append(np.stack(batch_probs, axis=0))
    preds = np.concatenate(all_preds, axis=1)
    probs = np.concatenate(all_probs, axis=1)
    labels = np.concatenate(all_labels)
    N = preds.shape[1]

    accs = [(preds[k] == labels).mean() * 100 for k in range(K)]
    agreements = []
    for i in range(K):
        for j in range(i + 1, K):
            agreements.append((preds[i] == preds[j]).mean() * 100)
    unanimous = np.all(preds == preds[0:1], axis=0).mean() * 100

    majority_preds = np.array([np.bincount(preds[:, n], minlength=100).argmax() for n in range(N)])
    pred_class_probs = np.array([[probs[k, n, majority_preds[n]] for n in range(N)] for k in range(K)])
    score_std = pred_class_probs.std(axis=0).mean()

    for s in students:
        s.train()
    return {
        'accs': accs,
        'acc_mean': float(np.mean(accs)),
        'acc_std': float(np.std(accs)),
        'pairwise_agreement_mean': float(np.mean(agreements)),
        'pairwise_agreement_std': float(np.std(agreements)),
        'unanimous_agreement': float(unanimous),
        'score_std_mean': float(score_std),
    }


def measure_transfer(students, testloader, device, num_images=500):
    K = len(students)
    for s in students:
        s.eval()
    correct_mask = None
    with torch.no_grad():
        all_correct = []
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            batch_correct = torch.ones(labels.size(0), dtype=torch.bool, device=device)
            for s in students:
                batch_correct &= (s(images).argmax(1) == labels)
            all_correct.append(batch_correct.cpu())
        correct_mask = torch.cat(all_correct)
    correct_indices = correct_mask.nonzero(as_tuple=False).squeeze().numpy()
    np.random.seed(42)
    if len(correct_indices) > num_images:
        correct_indices = np.random.choice(correct_indices, num_images, replace=False)

    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR100(root=DATA_ROOT, train=False, download=False, transform=transform_test)
    subset = Subset(testset, correct_indices)
    evalloader = DataLoader(subset, batch_size=64, shuffle=False)

    transfer_matrix = np.zeros((K, K))
    for src in range(K):
        all_adv, all_labels_list = [], []
        for images, labels in evalloader:
            images, labels = images.to(device), labels.to(device)
            adv = pgd_attack(students[src], images, labels, EPSILON, PGD_ALPHA, PGD_STEPS)
            all_adv.append(adv)
            all_labels_list.append(labels)
        all_adv = torch.cat(all_adv)
        all_labels_t = torch.cat(all_labels_list)
        N = all_adv.size(0)
        for tgt in range(K):
            with torch.no_grad():
                misclassified = 0
                for start in range(0, N, 64):
                    end = min(start + 64, N)
                    preds = students[tgt](all_adv[start:end]).argmax(1)
                    misclassified += (preds != all_labels_t[start:end]).sum().item()
                transfer_matrix[src][tgt] = misclassified / N * 100

    self_rates = [transfer_matrix[i][i] for i in range(K)]
    cross_rates = [transfer_matrix[i][j] for i in range(K) for j in range(K) if i != j]
    return {
        'transfer_matrix': transfer_matrix.tolist(),
        'self_asr': float(np.mean(self_rates)),
        'cross_asr': float(np.mean(cross_rates)),
        'transfer_ratio': float(np.mean(cross_rates) / max(np.mean(self_rates), 1e-10) * 100),
        'num_images': len(correct_indices),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda_", type=float, default=0.3)
    parser.add_argument("--T", type=float, default=4.0)
    parser.add_argument("--kl_target", type=float, default=5.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_students", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tag", type=str, default="v5b_default")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    K = args.num_students
    print(f"[*] Device: {device}")
    print(f"[*] All-pairs repulsive Divergent KD")
    print(f"[*] Hyperparams: alpha={args.alpha}, beta={args.beta}, lambda={args.lambda_}, "
          f"T={args.T}, kl_target={args.kl_target}")
    print(f"[*] Warmup: {args.warmup_epochs} epochs, Students: {K}, "
          f"Total epochs: {args.epochs}", flush=True)

    teacher = load_teacher(device)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR100(root=DATA_ROOT, train=True, download=False, transform=transform_train)
    testset = datasets.CIFAR100(root=DATA_ROOT, train=False, download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    students, optimizers, schedulers = [], [], []
    for i in range(K):
        s = create_student(device, seed=300 + i)
        s.train()
        opt = torch.optim.SGD(s.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        students.append(s)
        optimizers.append(opt)
        schedulers.append(sched)

    epoch_metrics = []
    t0 = time.time()

    for epoch in range(args.epochs):
        is_warmup = epoch < args.warmup_epochs
        phase_str = "WARMUP" if is_warmup else "DIVERGE"

        trainloader = DataLoader(trainset, batch_size=128, shuffle=True,
                                 num_workers=2, pin_memory=True)
        epoch_losses = [0.0] * K
        num_batches = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(images)

            if is_warmup:
                for k in range(K):
                    logits = students[k](images)
                    kl = F.kl_div(
                        F.log_softmax(logits / args.T, dim=1),
                        F.softmax(teacher_logits / args.T, dim=1),
                        reduction='batchmean') * (args.T * args.T)
                    ce = F.cross_entropy(logits, labels)
                    loss = args.alpha * ce + args.lambda_ * kl
                    optimizers[k].zero_grad()
                    loss.backward()
                    optimizers[k].step()
                    epoch_losses[k] += loss.item()
            else:
                all_logits = []
                for k in range(K):
                    all_logits.append(students[k](images))

                all_softmax = []
                with torch.no_grad():
                    for k in range(K):
                        all_softmax.append(F.softmax(all_logits[k].detach() / args.T, dim=1))

                for k in range(K):
                    ce = F.cross_entropy(all_logits[k], labels)
                    log_s_k = F.log_softmax(all_logits[k] / args.T, dim=1)

                    kl_teacher = F.kl_div(
                        log_s_k,
                        F.softmax(teacher_logits / args.T, dim=1),
                        reduction='batchmean') * (args.T * args.T)

                    repulsion = torch.tensor(0.0, device=device)
                    for j in range(K):
                        if j == k:
                            continue
                        kl_kj = F.kl_div(
                            log_s_k,
                            all_softmax[j],
                            reduction='none').sum(dim=1) * (args.T * args.T)
                        repulsion = repulsion + F.relu(args.kl_target - kl_kj).mean()
                    repulsion = repulsion / (K - 1)

                    loss = args.alpha * ce + args.lambda_ * kl_teacher + args.beta * repulsion
                    optimizers[k].zero_grad()
                    loss.backward()
                    optimizers[k].step()
                    epoch_losses[k] += loss.item()

            num_batches += 1

        for k in range(K):
            schedulers[k].step()

        avg_loss = np.mean([l / num_batches for l in epoch_losses])
        metrics = evaluate_students(students, testloader, device)
        metrics['epoch'] = epoch + 1
        metrics['avg_loss'] = float(avg_loss)
        metrics['phase'] = phase_str
        epoch_metrics.append(metrics)

        elapsed = time.time() - t0
        print(f"[Epoch {epoch+1:2d}/{args.epochs}] [{phase_str:7s}] "
              f"Loss={avg_loss:.4f} | "
              f"Acc={metrics['acc_mean']:.1f}+/-{metrics['acc_std']:.1f}% | "
              f"Agree={metrics['pairwise_agreement_mean']:.1f}% | "
              f"Unan={metrics['unanimous_agreement']:.1f}% | "
              f"ScoreStd={metrics['score_std_mean']:.4f} | "
              f"{elapsed:.0f}s", flush=True)

    print(f"\n[*] Measuring adversarial transferability...", flush=True)
    transfer = measure_transfer(students, testloader, device)
    print(f"[*] Self-attack: {transfer['self_asr']:.1f}% | "
          f"Cross-transfer: {transfer['cross_asr']:.1f}% | "
          f"Transfer ratio: {transfer['transfer_ratio']:.1f}%", flush=True)

    results = {
        'method': 'allpairs_repulsive_divergent_kd_v5b',
        'hyperparams': {
            'alpha': args.alpha, 'beta': args.beta, 'lambda': args.lambda_,
            'T': args.T, 'kl_target': args.kl_target, 'lr': args.lr,
            'warmup_epochs': args.warmup_epochs,
            'num_students': K, 'epochs': args.epochs,
        },
        'epoch_metrics': epoch_metrics,
        'transfer': transfer,
        'final_acc': epoch_metrics[-1]['acc_mean'],
        'final_agreement': epoch_metrics[-1]['pairwise_agreement_mean'],
        'final_unanimous': epoch_metrics[-1]['unanimous_agreement'],
    }
    tag = args.tag
    save_path = os.path.join(OUT_DIR, f"divergent_kd_results_{tag}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] Saved to {save_path}")

    ckpt_dir = os.path.join(CHECKPOINT_DIR, f"divergent_kd_{tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    for k, s in enumerate(students):
        ckpt_path = os.path.join(ckpt_dir, f"student_{k}.pth")
        torch.save(s.model.state_dict(), ckpt_path)
    print(f"[*] Saved {K} student checkpoints to {ckpt_dir}")
    print(f"[*] Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
