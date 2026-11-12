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
        for (unsigned n : {1u, 2u, 3u, 4u, 5u, 6u, 8u, 9u, 10u, 12u}) {
            const Index size = Index{1} << n;
            std::vector<double2> input(size), result(size), baseline(size);
            for (Index i = 0; i < size; ++i)
                input[i] = make_double2(std::sin(double(i)), std::cos(double(i)));
            double2* state = nullptr;
            check(cudaMalloc(reinterpret_cast<void**>(&state), size * sizeof(double2)));
            for (unsigned block : {64u, 128u, 256u}) {
                for (unsigned cap : {1u, 3u, 2147483647u}) {
                    for (unsigned q = 0; q < n; ++q) {
                        for (bool optimized : {false, true}) {
                            if (optimized && q != 0 && q != 8) continue;
                            const bool shuffle = optimized && q == 0 && n >= 5;
                            check(cudaMemcpy(state, input.data(), size * sizeof(double2), cudaMemcpyHostToDevice));
                            if (shuffle)
                                hadamardQubitZero<<<gridFor(size, block, cap), block>>>(state, size);
                            else if (useQ8PairSwizzle(optimized, q, n))
                                hadamard<true><<<gridFor(size / 2, block, cap), block>>>(state, size / 2, q);
                            else
                                hadamard<<<gridFor(size / 2, block, cap), block>>>(state, size / 2, q);
                            check(cudaGetLastError());
                            check(cudaMemcpy(result.data(), state, size * sizeof(double2), cudaMemcpyDeviceToHost));
                            if (!optimized) baseline = result;
                            // Independent amplitude-based reference checks both signs and imaginary parts.
                            for (Index i = 0; i < size; ++i) {
                                if (optimized && (std::abs(result[i].x - baseline[i].x) > 1e-12 ||
                                                std::abs(result[i].y - baseline[i].y) > 1e-12))
                                    throw std::runtime_error("Optimized/baseline mismatch");
                                if (!std::isfinite(result[i].x) || !std::isfinite(result[i].y))
                                    throw std::runtime_error("Non-finite Hadamard output");
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
                    // Full circuit from |0>, in both supported gate orders.
                    for (bool descending : {false, true}) {
                        check(cudaMemset(state, 0, size * sizeof(double2)));
                        const double2 one = make_double2(1.0, 0.0);
                        check(cudaMemcpy(state, &one, sizeof(one), cudaMemcpyHostToDevice));
                        for (unsigned gate = 0; gate < n; ++gate) {
                            const unsigned q = descending ? n - 1 - gate : gate;
                            if (q == 0 && n >= 5)
                                hadamardQubitZero<<<gridFor(size, block, cap), block>>>(state, size);
                            else if (useQ8PairSwizzle(true, q, n))
                                hadamard<true><<<gridFor(size / 2, block, cap), block>>>(state, size / 2, q);
                            else
                                hadamard<<<gridFor(size / 2, block, cap), block>>>(state, size / 2, q);
                            check(cudaGetLastError());
                        }
                        check(cudaMemcpy(result.data(), state, size * sizeof(double2), cudaMemcpyDeviceToHost));
                        const double expected = std::exp2(-0.5 * n);
                        for (const auto value : result)
                            if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
                                std::abs(value.x - expected) > expected * 1e-12 ||
                                std::abs(value.y) > expected * 1e-12)
                                throw std::runtime_error("Full circuit mismatch");
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
