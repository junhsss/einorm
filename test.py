import torch

from einorm import Einorm

m = Einorm("a b c", "b", b=100, bias=True)

x = torch.randn(1, 100, 4)

print(m(x).shape)
print(x)
print(m(x)[0, :, 0].mean())
print(sum(p.numel() for p in m.parameters()))
