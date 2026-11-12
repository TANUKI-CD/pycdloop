"""Type stubs for the compiled kernels."""

from typing import Any

def transform(buffer: Any, dimension: int, lanes: int, /) -> None: ...
def transform_mod(
    buffer: Any, dimension: int, lanes: int, modulus: int, interval: int = 30, /
) -> None: ...
def split(
    loop: Any, spectral: Any, signed: Any, dimension: int, lanes: int, /
) -> None: ...
def unsplit(
    spectral: Any, signed: Any, loop: Any, dimension: int, lanes: int, /
) -> None: ...
def split_mod(
    loop: Any, spectral: Any, signed: Any, dimension: int, lanes: int, modulus: int, /
) -> None: ...
def unsplit_mod(
    spectral: Any,
    signed: Any,
    loop: Any,
    dimension: int,
    lanes: int,
    modulus: int,
    inverse: int,
    /,
) -> None: ...
