# sanity_check.py
import torch
import time

print("=== SANITY CHECK ===")
print("Python:", __import__("sys").version.splitlines()[0])
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f" - Device {i}: {torch.cuda.get_device_name(i)}")
    # small tensor op on GPU
    t0 = torch.randn(1024, 1024, device="cuda")
    t1 = torch.randn(1024, 1024, device="cuda")
    s = time.time()
    out = torch.matmul(t0, t1)
    torch.cuda.synchronize()
    elapsed = time.time() - s
    print(f"Matrix multiply (1024x1024) took {elapsed:.4f} sec on GPU")
    # tiny model forward pass
    from torch import nn

    model = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 10)).cuda()
    x = torch.randn(2, 1024, device="cuda")
    y = model(x)
    print("Tiny model forward produced shape:", y.shape)
else:
    print("CUDA not available — check NVIDIA driver / docker runtime.")
print("=== END ===")
