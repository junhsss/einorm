# einorm

[![Test](https://github.com/junhsss/einorm/actions/workflows/test.yml/badge.svg)](https://github.com/junhsss/einorm/actions/workflows/test.yml)
[![PyPI Version](https://badge.fury.io/py/einorm.svg)](https://badge.fury.io/py/einorm)

An [einops](https://github.com/arogozhnikov/einops)-style generalized normalization layer.

## Installation

You need `torch` >= 1.13 or `functorch` to be installed:

```
pip install einorm
```

## Usage

```python
from einorm import Einorm

# Equivalent to nn.LayerNorm(1024)
Einorm("b n d", "d", d=1024)

# Specify the dimensions and sizes to normalize along.
Einorm("a b c d e", "b d", b=3, d=4)
```

According to [ViT-22B](https://arxiv.org/abs/2302.05442), normalizing query and key in a head-wise fashion can help stabilize the training dynamics. This can be achieved by providing additional grouping arguments to `Einorm`:

```python
Einorm("b h n d", "d", "h", h=16, d=64)  # num_heads=16, head_dim=64
```
