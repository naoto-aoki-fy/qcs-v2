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
            std::vector<double2> input(size);
            std::vector<double> inputReal(size), inputImag(size), resultReal(size), resultImag(size);
            for (Index i = 0; i < size; ++i) {
                input[i] = make_double2(std::sin(double(i)), std::cos(double(i)));
                inputReal[i] = input[i].x;
                inputImag[i] = input[i].y;
            }
            double* stateReal = nullptr;
            double* stateImag = nullptr;
            const auto componentBytes = size * sizeof(double);
            check(cudaMalloc(reinterpret_cast<void**>(&stateReal), componentBytes));
            check(cudaMalloc(reinterpret_cast<void**>(&stateImag), componentBytes));
            for (unsigned block : {64u, 128u, 256u}) {
                for (unsigned cap : {1u, 3u, 2147483647u}) {
                    for (unsigned q = 0; q < n; ++q) {
                        check(cudaMemcpy(stateReal, inputReal.data(), componentBytes, cudaMemcpyHostToDevice));
                        check(cudaMemcpy(stateImag, inputImag.data(), componentBytes, cudaMemcpyHostToDevice));
                        hadamard<<<gridFor(size / 2, block, cap), block>>>(stateReal, stateImag, size / 2, q);
                        check(cudaGetLastError());
                        check(cudaMemcpy(resultReal.data(), stateReal, componentBytes, cudaMemcpyDeviceToHost));
                        check(cudaMemcpy(resultImag.data(), stateImag, componentBytes, cudaMemcpyDeviceToHost));
                        // Independent amplitude-based reference checks both signs and imaginary parts.
                        for (Index i = 0; i < size; ++i) {
                            const Index bit = Index{1} << q;
                            const auto a = input[i & ~bit];
                            const auto b = input[i | bit];
                            const double sign = (i & bit) ? -1.0 : 1.0;
                            if (std::abs(resultReal[i] - (a.x + sign * b.x) / std::sqrt(2.0)) > 1e-12 ||
                                std::abs(resultImag[i] - (a.y + sign * b.y) / std::sqrt(2.0)) > 1e-12)
                                throw std::runtime_error("Hadamard reference mismatch");
                        }
                    }
                }
            }
            check(cudaFree(stateImag));
            check(cudaFree(stateReal));
        }
        std::cout << "CUDA Hadamard reference tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
