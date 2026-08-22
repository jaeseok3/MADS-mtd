"""Shared utilities for the TinyImageNet (200-class, 64x64) replication of the
CIFAR-100 MTD experiments. Reuses the dataset-agnostic black-box attacks from
eval_switching. Provides: TinyImageNet ImageFolder loaders, a ResNet-18 factory
(num_classes=200, CIFAR-style stem works on 64x64 via adaptive pooling),
normalization wrapper, and a pool loader."""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.resnet18_cifar100 import ResNet18_CIFAR100
import eval_switching as E  # dataset-agnostic attacks + clip_eps

DATA_ROOT = "data"
TIN_ROOT = os.path.join(DATA_ROOT, "tiny-imagenet-200")
CHECKPOINT_DIR = "checkpoints"
NUM_CLASSES = 200
# Standard TinyImageNet normalization statistics.
TIN_MEAN = (0.4802, 0.4481, 0.3975)
TIN_STD = (0.2770, 0.2691, 0.2821)
EPSILON = 8.0 / 255.0


class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def make_model(num_classes=NUM_CLASSES, arch='resnet'):
    if arch == 'resnet':
        return ResNet18_CIFAR100(num_classes=num_classes)
    import arch_common as _A
    return _A.make_model(arch, num_classes)


def wrap(model, device, eval_mode=True):
    w = NormalizedModel(model, TIN_MEAN, TIN_STD)
    w.to(device)
    if eval_mode:
        w.eval()
        for p in w.parameters():
            p.requires_grad = False
    return w


def train_transform():
    return transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def test_transform():
    return transforms.Compose([transforms.ToTensor()])


def trainset(transform=None):
    return datasets.ImageFolder(os.path.join(TIN_ROOT, "train"),
                                transform=transform or test_transform())


def valset(transform=None):
    # val/ is reorganized into class subfolders by tin_prepare.py
    return datasets.ImageFolder(os.path.join(TIN_ROOT, "val_organized"),
                                transform=transform or test_transform())


def load_teacher(device, arch='resnet'):
    m = make_model(arch=arch)
    name = "resnet18_tin_teacher.pth" if arch == 'resnet' else f"{arch}_tin_teacher.pth"
    m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, name), map_location="cpu", weights_only=True))
    return wrap(m, device)


def load_pool(pool_dir, device, arch='resnet'):
    ckpt_dir = os.path.join(CHECKPOINT_DIR, pool_dir)
    models = []
    for fname in sorted(os.listdir(ckpt_dir)):
        if fname.endswith('.pth'):
            m = make_model(arch=arch)
            m.load_state_dict(torch.load(os.path.join(ckpt_dir, fname),
                                         map_location="cpu", weights_only=True))
            models.append(wrap(m, device))
    return models


# ---- Fast in-memory cached dataset (built by tin_cache.py) ----
def load_cache(split):
    d = torch.load(os.path.join(DATA_ROOT, f"tin_cache_{split}.pt"))
    return d["x"], d["y"]  # uint8 [N,3,64,64], int64 [N]


def _gpu_random_crop(x, pad=8):
    # x: float [B,3,64,64] on device; pad-reflect then per-sample random 64x64 crop
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    ty = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    tx = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    ar = torch.arange(H, device=x.device)
    h_idx = (ty[:, None] + ar[None, :])  # [B,H]
    w_idx = (tx[:, None] + ar[None, :])  # [B,W]
    b = torch.arange(B, device=x.device)[:, None, None, None]
    c = torch.arange(C, device=x.device)[None, :, None, None]
    hh = h_idx[:, None, :, None]
    ww = w_idx[:, None, None, :]
    return xp[b, c, hh, ww]


class GPULoader:
    """Yields (x_float[0,1] on device, y on device) batches from cached uint8
    tensors, with optional GPU-side train augmentation (random crop + hflip)."""
    def __init__(self, x_u8, y, device, batch_size=128, shuffle=True, augment=False, drop_last=False):
        self.x = x_u8; self.y = y; self.device = device
        self.bs = batch_size; self.shuffle = shuffle; self.augment = augment
        self.drop_last = drop_last; self.N = x_u8.shape[0]

    def __len__(self):
        return self.N // self.bs if self.drop_last else (self.N + self.bs - 1) // self.bs

    def __iter__(self):
        order = torch.randperm(self.N) if self.shuffle else torch.arange(self.N)
        last = self.N - (self.N % self.bs) if self.drop_last else self.N
        for s in range(0, last, self.bs):
            idx = order[s:s + self.bs]
            xb = self.x[idx].to(self.device, non_blocking=True).float().div_(255.0)
            yb = self.y[idx].to(self.device, non_blocking=True)
            if self.augment:
                xb = _gpu_random_crop(xb, pad=8)
                flip = torch.rand(xb.shape[0], device=self.device) < 0.5
                xb[flip] = torch.flip(xb[flip], dims=[3])
            yield xb, yb


# Re-export dataset-agnostic attacks (operate on raw [0,1] C,H,W arrays).
square_attack = E.square_attack
simba_dct_attack = E.simba_dct_attack
signhunter_attack = E.signhunter_attack
clip_eps = E.clip_eps
