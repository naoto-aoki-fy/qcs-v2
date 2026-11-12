#pragma once

#include <cstdint>
#include <stdexcept>

using Index = std::uint64_t;

#ifdef __CUDACC__
#define QCS_HD __host__ __device__
#else
#define QCS_HD
#endif

// Insert a zero at the target bit. Valid for pair < 2^58 and bit <= 2^58.
QCS_HD inline Index firstAmplitude(Index pair, Index bit) {
    const Index low = pair & (bit - 1);
    return (pair << 1) - low;
}
// Swap pair bits 5 and 8; a permutation for power-of-two pair counts >= 512.
QCS_HD inline Index swizzleQ8Pair(Index pair) {
    const Index diff = ((pair >> 5) ^ (pair >> 8)) & Index{1};
    return pair ^ (diff << 5) ^ (diff << 8);
}

inline bool useQ8PairSwizzle(bool enabled, unsigned qubit, unsigned qubits) {
    return enabled && qubit == 8 && qubits >= 10;
}
#undef QCS_HD

inline unsigned int gridFor(Index pairs, unsigned int blockSize, unsigned int limit) {
    if (!pairs || !blockSize || !limit) throw std::invalid_argument("Invalid launch size");
    const Index blocks = pairs / blockSize + (pairs % blockSize != 0);
    return static_cast<unsigned int>(blocks < limit ? blocks : limit);
}
