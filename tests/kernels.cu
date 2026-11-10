#include "../src/kernels.cuh"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main() {
    try {
        for (unsigned n : {1u, 8u, 12u}) {
            const Index size = Index{1} << n;
            std::vector<double2> input(size), result(size);
            for (Index i = 0; i < size; ++i)
                input[i] = make_double2(std::sin(double(i)), std::cos(double(i)));
            double2* state = nullptr;
            check(cudaMalloc(reinterpret_cast<void**>(&state), size * sizeof(double2)));
            for (unsigned block : {64u, 128u, 256u}) {
                for (unsigned cap : {1u, 3u, 2147483647u}) {
                    for (unsigned q = 0; q < n; ++q) {
                        check(cudaMemcpy(state, input.data(), size * sizeof(double2), cudaMemcpyHostToDevice));
                        hadamard<<<gridFor(size / 2, block, cap), block>>>(state, size / 2, q);
                        check(cudaGetLastError());
                        check(cudaMemcpy(result.data(), state, size * sizeof(double2), cudaMemcpyDeviceToHost));
                        // Independent amplitude-based reference checks both signs and imaginary parts.
                        for (Index i = 0; i < size; ++i) {
                            const Index bit = Index{1} << q;
                            const auto a = input[i & ~bit];
                            const auto b = input[i | bit];
                            const double sign = (i & bit) ? -1.0 : 1.0;
                            if (std::abs(result[i].x - (a.x + sign * b.x) / std::sqrt(2.0)) > 1e-12 ||
                                std::abs(result[i].y - (a.y + sign * b.y) / std::sqrt(2.0)) > 1e-12)
                                throw std::runtime_error("Hadamard reference mismatch");
                        }
                    }
                }
            }
            check(cudaFree(state));
        }
        std::cout << "CUDA Hadamard reference tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
