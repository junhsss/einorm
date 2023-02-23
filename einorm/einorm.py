from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter, init
from torch.nn.functional import layer_norm

if hasattr(torch, "vmap"):
    from torch import vmap  # type: ignore
else:
    try:
        from functorch import vmap  # type: ignore
    except ImportError:
        raise ImportError(
            "einorm requires Torch version 1.13 or higher, "
            "or for Functorch to be installed."
        )


class EinormError(RuntimeError):
    """Runtime error thrown by einorm"""

    pass


class Einorm(Module):
    def __init__(
        self,
        pattern: str,
        target: str,
        group: Optional[str] = None,
        bias: bool = True,
        eps: float = 1e-5,
        device: Optional[Union[torch.device, str, None]] = None,
        dtype: Optional[torch.device] = None,
        **axes_length,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self._bias = bias

        _pattern: List[str] = pattern.split()
        _target: List[str] = target.split()
        _group: List[str] = group.split() if group else []

        _set_pattern = set(_pattern)
        _set_target = set(_target)
        _set_group = set(_group)

        if not _set_pattern:
            raise EinormError("Pattern expression does not contain any dimension.")

        if not _set_target:
            raise EinormError("Target expression does not contain any dimension.")

        if group and (not _set_group):
            raise EinormError("Group expression does not contain any dimension.")

        if len(_set_pattern) != len(_pattern):
            raise EinormError("Pattern expression contains duplicate dimension")

        if len(_set_target) != len(_target):
            raise EinormError("Target expression contains duplicate dimension")

        if len(_set_group) != len(_group):
            raise EinormError("Group expression contains duplicate dimension")

        _unknown = _set_target - _set_pattern
        if _unknown:
            raise EinormError(
                f"Target expression contains unknown dimension: {_unknown}"
            )

        _unknown = _set_group - _set_pattern
        if _unknown:
            raise EinormError(
                f"Group expression contains unknown dimension: {_unknown}"
            )

        _intersection = set.intersection(_set_target, _set_group)
        if _intersection:
            raise EinormError(
                f"Group and Target expression contain same dimension: {_intersection}"
            )

        _perm_dims = (
            ([_pattern.index(x) for x in _group] if group else [])
            + [
                i
                for i, x in enumerate(_pattern)
                if x not in _target + (_group if group else [])
            ]
            + [_pattern.index(x) for x in _target]
        )

        _perm = torch.tensor(
            _perm_dims,
            dtype=torch.long,
        )

        self.perm = _perm

        _inv = torch.empty_like(_perm)
        _inv[_perm] = torch.arange(_perm.size(0))

        self.inv = _inv

        # skip permute whenever possible
        self.skip_perm = _perm_dims == list(range(len(_pattern)))

        group_shape: Tuple[int, ...] = ()
        if group:
            for axis in _group:
                if axis in axes_length:
                    if isinstance(axes_length[axis], int):
                        group_shape += (axes_length[axis],)
                    else:
                        raise EinormError(f"Size must be integer for axis {axis}")
                else:
                    raise EinormError(f"Specify size for axis {axis}")

        target_shape: Tuple[int, ...] = ()  # type: ignore
        for axis in _target:
            if axis in axes_length:
                if isinstance(axes_length[axis], int):
                    target_shape += (axes_length[axis],)
                else:
                    raise EinormError(f"Size must be integer for axis {axis}")
            else:
                raise EinormError(f"Specify size for axis {axis}")

        self.weight = Parameter(
            torch.empty(group_shape + target_shape, **factory_kwargs)
        )

        if bias:
            self.bias = Parameter(
                torch.empty(group_shape + target_shape, **factory_kwargs),
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self._normalizer = self.construct_normalizer(target_shape, group_shape, eps)

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self._bias:
            init.zeros_(self.bias)

    # FIXME: ditch permute
    def forward(self, x: Tensor) -> Tensor:
        if self.skip_perm:
            return self._normalizer(x, self.weight, self.bias)

        x = x.permute(*self.perm)  # type: ignore
        x = self._normalizer(x, self.weight, self.bias)
        x = x.permute(*self.inv)  # type: ignore
        return x

    def construct_normalizer(
        self,
        target_shape: Tuple[int, ...],
        group_shape: Tuple[int, ...],
        eps: float,
    ) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
        def _normalizer(
            input: Tensor,
            weight: Tensor,
            bias: Union[Tensor, None],
        ) -> Tensor:
            return layer_norm(
                input,
                normalized_shape=target_shape,
                weight=weight,
                bias=bias,
                eps=eps,
            )

        if group_shape:
            len_group_shape = len(group_shape)

            _dims = (
                0 if len_group_shape == 1 else tuple(i for i in range(len_group_shape))
            )
            return vmap(
                _normalizer,
                in_dims=_dims,
                out_dims=_dims,
            )

        return _normalizer
