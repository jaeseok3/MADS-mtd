"""Architecture- and dataset-parametrized utilities for the multi-architecture
(ResNet-18 / VGG-16-BN / AlexNet) x (CIFAR-100 / TinyImageNet-200) campaign.
VGG is wrapped with adaptive pooling so the same weights work at 32x32 and 64x64.
Reuses the dataset-agnostic black-box attacks from eval_switching."""
import os, sys
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.resnet18_cifar100 import ResNet18_CIFAR100
from models.vgg16_cifar100 import VGG16_CIFAR100
from models.alexnet_cifar100 import AlexNet_CIFAR100
import eval_switching as E

DATA_ROOT = "data"
CHECKPOINT_DIR = "checkpoints"
EPSILON = 8.0 / 255.0
NORM = {'cifar': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        'tin':   ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        'c10':   ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))}
NUMCLASSES = {'cifar': 100, 'tin': 200, 'c10': 10}


class VGGAny(VGG16_CIFAR100):
    """VGG-16-BN with adaptive average pooling -> works at any input resolution;
    weight-compatible with the 32x32 CIFAR checkpoints (32x32 features are 1x1)."""
    def forward(self, x):
        x = self.features(x); x = F.adaptive_avg_pool2d(x, 1)
        return self.fc(x.view(x.size(0), -1))


def make_model(arch, num_classes):
    if arch == 'resnet':  return ResNet18_CIFAR100(num_classes)
    if arch == 'vgg':     return VGGAny(num_classes)
    if arch == 'alexnet': return AlexNet_CIFAR100(num_classes)
    raise ValueError(arch)


class NormalizedModel(nn.Module):
    def __init__(s, m, mean, std):
        super().__init__(); s.model = m
        s.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        s.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    def forward(s, x): return s.model((x - s.mean) / s.std)


def wrap(model, dataset, device, eval_mode=True):
    w = NormalizedModel(model, *NORM[dataset]); w.to(device)
    if eval_mode:
        w.eval()
        for p in w.parameters(): p.requires_grad = False
    return w


def load_pool(tag, arch, dataset, device):
    d = os.path.join(CHECKPOINT_DIR, tag); models = []
    for f in sorted(os.listdir(d)):
        if f.endswith('.pth'):
            m = make_model(arch, NUMCLASSES[dataset])
            m.load_state_dict(torch.load(os.path.join(d, f), map_location='cpu', weights_only=True))
            models.append(wrap(m, dataset, device))
    return models


def load_teacher(ckpt, arch, dataset, device):
    m = make_model(arch, NUMCLASSES[dataset])
    m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, ckpt), map_location='cpu', weights_only=True))
    return wrap(m, dataset, device)


# ---- dataset access for evaluation (indexable img[i] -> float CHW [0,1], label) ----
class _TIN:
    def __init__(s):
        d = torch.load(os.path.join(DATA_ROOT, "tin_cache_val.pt"))
        s.x = d["x"]; s.y = d["y"]
    def __len__(s): return s.x.shape[0]
    def __getitem__(s, i): return s.x[i].float() / 255.0, int(s.y[i])

def get_testset(dataset):
    if dataset == 'tin': return _TIN()
    from torchvision import datasets, transforms
    if dataset == 'c10':
        return datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transforms.ToTensor())
    return datasets.CIFAR100(root=DATA_ROOT, train=False, download=False, transform=transforms.ToTensor())


def img_np(testset, i):
    x, y = testset[int(i)]
    return (x.numpy() if isinstance(x, torch.Tensor) else x), int(y)


def correct_by_all(pool, testset, device, bs=256):
    N = len(testset); idx = []
    xs = torch.stack([testset[i][0] if isinstance(testset[i][0], torch.Tensor)
                      else torch.from_numpy(testset[i][0]) for i in range(min(N, 0))]) if False else None
    # batched correctness
    labs = np.array([testset[i][1] for i in range(N)])
    correct = np.ones(N, dtype=bool)
    with torch.no_grad():
        for s0 in range(0, N, bs):
            e = min(s0+bs, N)
            xb = torch.stack([(testset[i][0] if isinstance(testset[i][0], torch.Tensor)
                               else torch.from_numpy(testset[i][0])).float() for i in range(s0, e)]).to(device)
            ag = np.ones(e-s0, dtype=bool)
            for m in pool: ag &= (m(xb).argmax(1).cpu().numpy() == labs[s0:e])
            correct[s0:e] = ag
    return np.nonzero(correct)[0], labs


square_attack = E.square_attack
simba_dct_attack = E.simba_dct_attack
signhunter_attack = E.signhunter_attack
clip_eps = E.clip_eps
