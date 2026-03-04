#!/usr/bin/env python3
"""
Verify softmax decomposition against torch.softmax using allclose.

Decomposition:
  softmax(x, dim) = exp(x - max(x, dim, keepdim=True)) /
                    sum(exp(x - max(x, dim, keepdim=True)), dim, keepdim=True)
"""

import sys


def main() -> int:
    try:
        import torch
    except Exception:
        print("torch is not installed; skipping runtime decomposition check")
        return 0

    x = torch.randn(3, 5, dtype=torch.float32)
    dim = 1

    ref = torch.nn.Softmax(dim=dim)(x)
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    decomp = torch.exp(shifted) / torch.sum(torch.exp(shifted), dim=dim, keepdim=True)

    ok = torch.allclose(ref, decomp, rtol=1e-5, atol=1e-6)
    print(f"allclose={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
