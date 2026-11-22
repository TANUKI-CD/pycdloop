# Security policy

pycdloop is a numerical library with no network access, file I/O, or dynamic
code execution. Security-relevant issues are therefore incorrect results in
exact arithmetic, and memory-safety defects in its extension module
`pycdloop._core`: out-of-bounds reads or writes, integer overflow in size
computations, or reference-counting errors. Defects in the Cayley-Dickson
kernels of fastCD should be reported to that project.

The arithmetic is not constant-time: running time and memory access depend
on the values of the operands. The package is suitable for research on
cryptographic constructions, not for deploying them.
`LoopAlgebra.random(secure=True)` draws from the operating system's
cryptographic source; every other use of randomness is a NumPy generator,
which is reproducible from its seed and not suitable for secrets.

Please report issues privately to the maintainer through the contact details
on the canonical repository,
[git.tanuki-cd.com/algebraity/pycdloop](https://git.tanuki-cd.com/algebraity/pycdloop),
rather than in a public issue. Include the loop, the coefficient field, and
a minimal input that reproduces the problem. A fix will be released as a
patch version and noted in `CHANGELOG.md`.
