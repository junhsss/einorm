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

`einorm` provides `Einorm` module which is straightforward to use:

```python
from einorm import Einorm

# Specify the dimensions and sizes to normalize along.
Einorm("a b c d e", "b d", b=3, d=4)
```

According to [ViT-22B](https://arxiv.org/abs/2302.05442), normalizing query and key in a head-wise fashion stabilizes the training process. This can be achieved by providing additional arguments to `Einorm`:

```python
Einorm("b h n d", "h", "d", h=16, d=64)  # num_heads=16, head_dim=64
```
