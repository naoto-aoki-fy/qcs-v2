#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include "indexing.hpp"

__global__ void hadamard(double* stateReal, double* stateImag, 
                         Index pairs, unsigned int qubit) {
    const Index begin = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = static_cast<Index>(gridDim.x) * blockDim.x;
    const Index bit = Index{1} << qubit;
    for (Index pair = begin; pair < pairs; pair += stride) {
        const Index first = firstAmplitude(pair, bit);
        const Index second = first | bit;
        const double aReal = stateReal[first];
        const double aImag = stateImag[first];
        const double bReal = stateReal[second];
        const double bImag = stateImag[second];
        constexpr double scale = 0.7071067811865475244;
        stateReal[first] = (aReal + bReal) * scale;
        stateImag[first] = (aImag + bImag) * scale;
        stateReal[second] = (aReal - bReal) * scale;
        stateImag[second] = (aImag - bImag) * scale;
    }
}

__global__ void verify(const double* stateReal, const double* stateImag,
                       Index pairs, double expected,
                       unsigned int* invalid) {
    const Index begin = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = static_cast<Index>(gridDim.x) * blockDim.x;
    for (Index pair = begin; pair < pairs; pair += stride) {
        const double aReal = stateReal[2 * pair];
        const double aImag = stateImag[2 * pair];
        const double bReal = stateReal[2 * pair + 1];
        const double bImag = stateImag[2 * pair + 1];
        const double tolerance = expected * 1.0e-12;
        if (!isfinite(aReal) || !isfinite(aImag) || !isfinite(bReal) || !isfinite(bImag) ||
            fabs(aReal - expected) > tolerance || fabs(aImag) > tolerance ||
            fabs(bReal - expected) > tolerance || fabs(bImag) > tolerance) {
            atomicExch(invalid, 1u);
        }
    }
}
