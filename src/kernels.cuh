#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include "indexing.hpp"

// Requires a block size and amplitude count divisible by 32 (N >= 5).
__global__ void hadamardQubitZero(double2* state, Index amplitudes) {
    const Index begin = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = static_cast<Index>(gridDim.x) * blockDim.x;
    constexpr unsigned int mask = 0xffffffffu;
    constexpr double scale = 0.7071067811865475244;
    const Index lane = threadIdx.x & 31u;
    for (Index base = begin - lane; base < amplitudes; base += stride) {
        const Index index = base + lane;
        const double2 own = state[index];
        const double real = __shfl_xor_sync(mask, own.x, 1);
        const double imag = __shfl_xor_sync(mask, own.y, 1);
        const double2 result = (lane & 1u)
            ? make_double2((real - own.x) * scale, (imag - own.y) * scale)
            : make_double2((own.x + real) * scale, (own.y + imag) * scale);
        state[index] = result;
    }
}

__global__ void hadamard(double2* state, Index pairs, unsigned int qubit) {
    const Index begin = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = static_cast<Index>(gridDim.x) * blockDim.x;
    const Index bit = Index{1} << qubit;
    for (Index pair = begin; pair < pairs; pair += stride) {
        const Index first = firstAmplitude(pair, bit);
        const Index second = first | bit;
        const double2 a = state[first];
        const double2 b = state[second];
        constexpr double scale = 0.7071067811865475244;
        state[first] = make_double2((a.x + b.x) * scale, (a.y + b.y) * scale);
        state[second] = make_double2((a.x - b.x) * scale, (a.y - b.y) * scale);
    }
}

__global__ void verify(const double2* state, Index pairs, double expected,
                       unsigned int* invalid) {
    const Index begin = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = static_cast<Index>(gridDim.x) * blockDim.x;
    for (Index pair = begin; pair < pairs; pair += stride) {
        const double2 a = state[2 * pair];
        const double2 b = state[2 * pair + 1];
        const double tolerance = expected * 1.0e-12;
        if (!isfinite(a.x) || !isfinite(a.y) || !isfinite(b.x) || !isfinite(b.y) ||
            fabs(a.x - expected) > tolerance || fabs(a.y) > tolerance ||
            fabs(b.x - expected) > tolerance || fabs(b.y) > tolerance) {
            atomicExch(invalid, 1u);
        }
    }
}
