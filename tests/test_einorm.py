import pytest
import torch

from einorm import Einorm
from einorm.einorm import EinormError


def test_default():
    normalizer = Einorm("b c h w", target="c h w", c=3, h=224, w=224)
    x = torch.randn(32, 3, 224, 224)
    normalizer(x)


def test_multihead():
    normalizer = Einorm("b h n d", target="d", group="h", d=768, h=8)
    x = torch.randn(32, 8, 196, 768)
    normalizer(x)


def test_empty():
    normalizer = Einorm("b   ha n d   ", "ha d", "n", ha=8, n=196, d=100)
    x = torch.randn(32, 8, 196, 100)
    normalizer(x)


def test_error():
    with pytest.raises(EinormError):
        Einorm("a b c d", "a b")

    with pytest.raises(EinormError):
        Einorm("a b c d", "a b", "b")

    with pytest.raises(EinormError):
        Einorm("a b c d", "a d", "d b c")

    with pytest.raises(EinormError):
        Einorm("a b c d", "a d", "e g d")

    with pytest.raises(EinormError):
        Einorm("", "")

    with pytest.raises(EinormError):
        Einorm("a b c", "h", "")
