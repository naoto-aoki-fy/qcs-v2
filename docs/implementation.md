# Implementation and validation

The optional qubit count is an integer from 1 through 59; the default is 33.
N qubits have indices 0 through N−1, so exactly N Hadamard kernels are launched.
The upper input bound prevents overflow of the amplitude count and byte count;
actual feasibility is limited by available memory on CUDA device 0. No other GPU
is used, and no cuStateVec dependency is present.

Each complex amplitude is a `double2` (16 bytes). The initial state is
amplitude 1 at index zero and 0 everywhere else. Each Hadamard kernel uses 64
threads per block, and each active thread reads and writes exactly one pair of
complex amplitudes. Pair indices insert a zero bit at the target qubit to find
the first amplitude; the second has that bit set. All state indices and index
arithmetic use unsigned 64-bit integers. There is no gate fusion or grid-stride
loop. Large launches use a two-dimensional grid when necessary; boundary threads
in a partially filled grid return immediately.

CUDA events bracket only the sequence of Hadamard launches in the default
stream. Allocation, zeroing, initialization, and optional verification precede
or follow this interval. The reported measurement is one pass, without warmup
or averaging. As with CUDA event timing generally, any GPU idle gaps between
launches within that interval are included. API failures, launch failures, and
asynchronous execution failures are reported to standard error with a nonzero
exit status. The program rejects insufficient free memory before allocation;
allocation can still fail if available memory changes after the check.

With `--verify`, a separate kernel checks every final amplitude against
`2^(-N/2) + 0i`, using a relative tolerance of `1e-12`, and rejects non-finite
values. This check and its memory allocation are outside the measured interval.

On a CUDA-equipped machine, build as shown in the README and smoke-test:

```sh
./build/cuda_state_vector 1 --verify
./build/cuda_state_vector 8 --verify
./build/cuda_state_vector 20 --verify
./build/cuda_state_vector 33 --verify
```

The last command requires more than 128 GiB of available GPU memory including
runtime overhead. On a smaller GPU it must fail with an insufficient-memory
error. Invalid inputs such as `0`, `60`, and `abc` must also fail. For a device
memory error check, run `compute-sanitizer --tool memcheck` on the 8-qubit case.
