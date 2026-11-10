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
    return (pair & (bit - 1)) | ((pair & ~(bit - 1)) << 1);
}
#undef QCS_HD

inline unsigned int gridFor(Index pairs, unsigned int blockSize, unsigned int limit) {
    if (!pairs || !blockSize || !limit) throw std::invalid_argument("Invalid launch size");
    const Index blocks = pairs / blockSize + (pairs % blockSize != 0);
    return static_cast<unsigned int>(blocks < limit ? blocks : limit);
}
