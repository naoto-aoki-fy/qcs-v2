# CUDA state vector simulator

Single-GPU CUDA C++ simulator using FP64 complex amplitudes. Defaults to 33
qubits initialized to the all-zero computational basis state, then applies one
Hadamard gate per qubit in ascending order (0 through N−1).

Requires GNU Make, a CUDA toolkit with C++17 support, and an NVIDIA GPU.
Generate `config.mk` with [nvccoptions](https://github.com/naoto-aoki-fy/nvccoptions)
and copy it to the project root.
The default state vector alone requires 128 GiB of free GPU memory.

```sh
make
./build/cuda_state_vector
./build/cuda_state_vector 20 --repetitions 128 --verify
```

Repeats the circuit 128 times by default (`--repetitions M` to override), printing
each sample's GPU time in milliseconds. Optional verification runs after each sample. See [implementation and validation](docs/implementation.md).

See [build configuration](docs/build.md) for setup and custom options.
