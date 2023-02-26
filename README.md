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

`nn.LayerNorm` is great, but it only normalizes tensors over the last few dimensions.
While this may be beneficial from a performance perspective, it often requires manual dimension rearrangement before it can be used. This is where [einops](https://github.com/arogozhnikov/einops) semantics come in handy.

The `Einorm` module can be used as a drop-in replacement for `nn.LayerNorm`. For example:

```python
from einorm import Einorm

# Equivalent to nn.LayerNorm(1024)
Einorm("b n d", "d", d=1024)
```

Of course, You can normalize over any dimensions you want:

```python
Einorm("a b c d e", "b d", b=12, d=34)
```

**Caveats:** `Einorm` internally depends on `nn.functional.layer_norm` anyway.
Therefore, if you are not normalizing over the last dimensions, `permute` and `contiguous` calls will happen, which may incur slight performance degradation.
If you are normalizing over the last dimensions, `Einorm` skips `permute` call, so the performance will be identical to `nn.LayerNorm`.

### Grouped Layer Normalization

According to [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442), normalizing query and key in a head-wise fashion can help stabilize the training dynamics.
However, since `nn.LayerNorm` only calculates the mean and standard-deviation over the last few dimensions and normalizes over those few dimensions using the same statistics, it can be tricky to implement these behaviors.

This can be achieved by providing additional grouping arguments to `Einorm`:

```python
Einorm("b h n d", "d", "h", h=16, d=64)  # num_heads=16, head_dim=64
```

Here, `Einorm` normalizes over the last dimension using per-head statistics and parameters.

It differs from the following, where `Einorm` normalizes over `h` and `d` dimensions using the same statistics and parameters:

```python
Einorm("b h n d", "h d", h=16, d=64)
```

`Einorm` uses `functorch.vmap` to support this behavior with optimal performance.
Therefore, you will need `functorch` or `torch` >=1.13, which natively supports `vmap`.

### Without bias

For some reason, `nn.LayerNorm` does not have an option for disabling bias.
You can [safely omit bias](https://arxiv.org/abs/2302.05442) using `bias` option.

```python
Einorm("b c h w", "h w", h=256, w=256, bias=False)
```
