"""Build configuration for the compiled kernels of pycdloop.

Package metadata lives in pyproject.toml. This file declares the extension
module ``pycdloop._core``, which holds the Walsh--Hadamard transform and the
change of coordinates between the loop basis and the split form.
"""

import sys

from setuptools import Extension, setup

if sys.platform == "win32":
    # The source is C11; the Microsoft compiler accepts ``restrict`` only when
    # that standard is requested
    compile_args = ["/O2", "/fp:precise", "/std:c11"]
else:
    compile_args = ["-O3", "-fno-fast-math", "-ffp-contract=off"]

setup(
    ext_modules=[
        Extension(
            "pycdloop._core",
            sources=["src/pycdloop/_core.c"],
            extra_compile_args=compile_args,
        )
    ]
)
