import sys

import accelerate
import datasets
import peft
import torch
import transformers
import trl
import wandb

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)

gpu_available = torch.backends.mps.is_available()
print("Mac GPU:", gpu_available)

if gpu_available:
    x = torch.tensor(
        [1.0, 2.0],
        device="mps",
        requires_grad=True,
    )
    loss = x.square().sum()
    loss.backward()
    print("GPUの勾配計算:", x.grad.cpu().tolist())