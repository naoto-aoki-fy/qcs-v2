# CUDA state vector simulator

Single-GPU CUDA C++ simulator using FP64 complex amplitudes. Defaults to 33
qubits initialized to the all-zero computational basis state, then applies one
Hadamard gate per qubit in ascending order (0 through N−1).

Requires CMake 3.24+, a CUDA toolkit with C++17 support, and an NVIDIA GPU.
The default state vector alone requires 128 GiB of free GPU memory.

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
./build/cuda_state_vector
./build/cuda_state_vector 20 --verify
```

Prints the total GPU time for the N Hadamard kernels in milliseconds. Optional
verification runs after timing. See [implementation and validation](docs/implementation.md).
