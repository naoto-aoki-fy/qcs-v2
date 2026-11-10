#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using Index = std::uint64_t;
constexpr unsigned int kBlockSize = 64;
static_assert(sizeof(double2) == 16, "Expected FP64 complex storage");
static_assert(sizeof(std::size_t) == 8, "A 64-bit host is required");

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
    }
}

struct Resources {
    cudaStream_t stream = nullptr;
    double2* state = nullptr;
    unsigned int* invalid = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    // Explicit checked cleanup on success; best-effort cleanup during unwinding.
    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        if (invalid) cudaFree(invalid);
        if (state) cudaFree(state);
        if (stream) cudaStreamDestroy(stream);
    }

    void release() {
        check(cudaStreamSynchronize(stream), "complete stream work");
        check(cudaEventDestroy(stop), "destroy stop event");
        stop = nullptr;
        check(cudaEventDestroy(start), "destroy start event");
        start = nullptr;
        if (invalid) {
            check(cudaFree(invalid), "free verification flag");
            invalid = nullptr;
        }
        check(cudaFree(state), "free state vector");
        state = nullptr;
        check(cudaStreamDestroy(stream), "destroy stream");
        stream = nullptr;
    }
};

__global__ void hadamard(double2* state, Index pairs, unsigned int qubit) {
    const Index block = static_cast<Index>(blockIdx.y) * gridDim.x + blockIdx.x;
    const Index pair = block * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;

    const Index bit = Index{1} << qubit;
    const Index low = pair & (bit - 1);
    const Index first = ((pair >> qubit) << (qubit + 1)) | low;
    const Index second = first | bit;
    const double2 a = state[first];
    const double2 b = state[second];
    constexpr double scale = 0.7071067811865475244;
    state[first] = make_double2((a.x + b.x) * scale, (a.y + b.y) * scale);
    state[second] = make_double2((a.x - b.x) * scale, (a.y - b.y) * scale);
}

__global__ void verify(const double2* state, Index pairs, double expected,
                       unsigned int* invalid) {
    const Index block = static_cast<Index>(blockIdx.y) * gridDim.x + blockIdx.x;
    const Index pair = block * blockDim.x + threadIdx.x;
    if (pair >= pairs) return;
    const double2 a = state[2 * pair];
    const double2 b = state[2 * pair + 1];
    const double tolerance = expected * 1.0e-12;
    if (!isfinite(a.x) || !isfinite(a.y) || !isfinite(b.x) || !isfinite(b.y) ||
        fabs(a.x - expected) > tolerance || fabs(a.y) > tolerance ||
        fabs(b.x - expected) > tolerance || fabs(b.y) > tolerance) {
        atomicExch(invalid, 1u);
    }
}

dim3 gridFor(Index pairs, const cudaDeviceProp& properties) {
    const Index blocks = (pairs + kBlockSize - 1) / kBlockSize;
    const Index x = blocks < static_cast<Index>(properties.maxGridSize[0])
                        ? blocks : static_cast<Index>(properties.maxGridSize[0]);
    const Index y = (blocks + x - 1) / x;
    if (y > static_cast<Index>(properties.maxGridSize[1])) {
        throw std::runtime_error("Required grid exceeds device launch limits");
    }
    return dim3(static_cast<unsigned int>(x), static_cast<unsigned int>(y));
}

int main(int argc, char** argv) {
    try {
        unsigned int qubits = 33;
        unsigned int repetitions = 128;
        bool repetitionsProvided = false;
        bool shouldVerify = false;
        bool countProvided = false;
        for (int i = 1; i < argc; ++i) {
            const std::string argument(argv[i]);
            if (argument == "--verify" && !shouldVerify) {
                shouldVerify = true;
            } else if (argument == "--repetitions" && !repetitionsProvided) {
                if (++i >= argc) {
                    throw std::runtime_error("--repetitions requires a positive integer");
                }
                const std::string count(argv[i]);
                if (count.empty() ||
                    count.find_first_not_of("0123456789") != std::string::npos) {
                    throw std::runtime_error("--repetitions requires a positive integer");
                }
                const unsigned long long value = std::stoull(count);
                if (value < 1 || value > std::numeric_limits<unsigned int>::max()) {
                    throw std::runtime_error("Repetition count is out of range");
                }
                repetitions = static_cast<unsigned int>(value);
                repetitionsProvided = true;
            } else if (!countProvided && !argument.empty() &&
                       argument.find_first_not_of("0123456789") == std::string::npos) {
                const unsigned long long value = std::stoull(argument);
                if (value < 1 || value > 59) {
                    throw std::runtime_error("Qubit count must be between 1 and 59");
                }
                qubits = static_cast<unsigned int>(value);
                countProvided = true;
            } else {
                throw std::runtime_error("Usage: cuda_state_vector [qubits] [--repetitions M] [--verify]");
            }
        }

        check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync),
              "enable blocking synchronization");
        check(cudaSetDevice(0), "select device 0");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0), "query device properties");
        if (properties.maxThreadsPerBlock < kBlockSize ||
            properties.maxThreadsDim[0] < kBlockSize) {
            throw std::runtime_error("Device cannot launch 64-thread blocks");
        }
        const Index amplitudes = Index{1} << qubits;
        const std::size_t bytes = amplitudes * sizeof(double2);
        const std::size_t extraBytes = shouldVerify ? sizeof(unsigned int) : 0;
        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;
        check(cudaMemGetInfo(&freeBytes, &totalBytes), "query GPU memory");
        if (bytes > freeBytes || extraBytes > freeBytes - bytes) {
            throw std::runtime_error("Insufficient GPU memory: need " +
                std::to_string(bytes + extraBytes) + " bytes, available " +
                std::to_string(freeBytes) + " bytes");
        }
        const Index pairs = amplitudes / 2;
        const dim3 grid = gridFor(pairs, properties);
        const double2 one = make_double2(1.0, 0.0);
        unsigned int invalid = 0;
        Resources resources;
        check(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking),
              "create execution stream");
        check(cudaMalloc(reinterpret_cast<void**>(&resources.state), bytes),
              "allocate state vector");
        check(cudaEventCreateWithFlags(&resources.start, cudaEventBlockingSync),
              "create start event");
        check(cudaEventCreateWithFlags(&resources.stop, cudaEventBlockingSync),
              "create stop event");
        if (shouldVerify) {
            check(cudaMalloc(reinterpret_cast<void**>(&resources.invalid),
                             sizeof(unsigned int)), "allocate verification flag");
        }
        for (unsigned int sample = 0; sample < repetitions; ++sample) {
            check(cudaMemsetAsync(resources.state, 0, bytes, resources.stream), "zero state vector");
            check(cudaMemcpyAsync(resources.state, &one, sizeof(one),
                                  cudaMemcpyHostToDevice, resources.stream),
                  "initialize amplitude zero");

            check(cudaEventRecord(resources.start, resources.stream), "record start event");
            for (unsigned int qubit = 0; qubit < qubits; ++qubit) {
                hadamard<<<grid, kBlockSize, 0, resources.stream>>>(resources.state, pairs, qubit);
                check(cudaGetLastError(), "launch Hadamard kernel");
            }
            check(cudaEventRecord(resources.stop, resources.stream), "record stop event");
            check(cudaEventSynchronize(resources.stop), "complete Hadamard kernels");
            float milliseconds = 0.0f;
            check(cudaEventElapsedTime(&milliseconds, resources.start, resources.stop),
                  "measure elapsed GPU time");

            if (shouldVerify) {
                check(cudaMemsetAsync(resources.invalid, 0, sizeof(unsigned int),
                                      resources.stream),
                      "reset verification flag");
                const double expected = std::exp2(-0.5 * qubits);
                verify<<<grid, kBlockSize, 0, resources.stream>>>(resources.state, pairs, expected,
                                            resources.invalid);
                check(cudaGetLastError(), "launch verification kernel");
                check(cudaMemcpyAsync(&invalid, resources.invalid, sizeof(invalid),
                                      cudaMemcpyDeviceToHost, resources.stream), "read verification result");
                check(cudaStreamSynchronize(resources.stream), "complete verification");
                if (invalid) throw std::runtime_error("State verification failed");
            }
            std::cout << "Sample " << (sample + 1) << "/" << repetitions
                      << " Hadamard kernels: " << std::fixed << std::setprecision(3)
                      << milliseconds << " ms\n";
        }
        resources.release();
        if (shouldVerify) std::cout << "Verification passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
