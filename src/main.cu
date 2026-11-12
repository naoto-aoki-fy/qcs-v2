#include "kernels.cuh"
#include "qubit_order.hpp"
#include "timing_csv.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
    std::vector<cudaEvent_t> gateEvents;

    // Explicit checked cleanup on success; best-effort cleanup during unwinding.
    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        for (auto event : gateEvents) {
            if (event) cudaEventDestroy(event);
        }
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        if (invalid) cudaFree(invalid);
        if (state) cudaFree(state);
        if (stream) cudaStreamDestroy(stream);
    }

    void release() {
        check(cudaStreamSynchronize(stream), "complete stream work");
        for (auto& event : gateEvents) {
            check(cudaEventDestroy(event), "destroy gate timing event");
            event = nullptr;
        }
        gateEvents.clear();
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

int main(int argc, char** argv) {
    try {
        unsigned int qubits = 33;
        unsigned int repetitions = 128;
        bool repetitionsProvided = false;
        unsigned int blockSize = kBlockSize;
        unsigned int warmup = 0;
        unsigned int maxBlocks = 0;
        bool shouldVerify = false;
        bool timeGates = false;
        bool baselineQubitZero = false;
        bool q8PairSwizzle = false;
        bool countProvided = false;
        QubitOrder qubitOrder = QubitOrder::Ascending;
        for (int i = 1; i < argc; ++i) {
            const std::string argument(argv[i]);
            if (argument == "--qubit-order") {
                if (++i >= argc)
                    throw std::runtime_error("--qubit-order requires ascending or descending");
                qubitOrder = parseQubitOrder(argv[i]);
            } else if (argument == "--block-size" || argument == "--warmup" ||
                argument == "--max-blocks") {
                if (++i >= argc) throw std::runtime_error(argument + " requires an integer");
                const std::string count(argv[i]);
                if (count.empty() || count.find_first_not_of("0123456789") != std::string::npos)
                    throw std::runtime_error(argument + " requires an integer");
                const auto value = std::stoull(count);
                if (value > std::numeric_limits<unsigned int>::max())
                    throw std::runtime_error(argument + " is out of range");
                if (argument == "--block-size") {
                    if (value != 64 && value != 128 && value != 256)
                        throw std::runtime_error("Block size must be 64, 128, or 256");
                    blockSize = static_cast<unsigned int>(value);
                } else if (argument == "--warmup") {
                    warmup = static_cast<unsigned int>(value);
                } else {
                    if (value == 0) throw std::runtime_error("--max-blocks must be positive");
                    maxBlocks = static_cast<unsigned int>(value);
                }
            } else if (argument == "--q8-pair-swizzle" && !q8PairSwizzle) {
                q8PairSwizzle = true;
            } else if (argument == "--baseline-q0" && !baselineQubitZero) {
                baselineQubitZero = true;
            } else if (argument == "--time-gates" && !timeGates) {
                timeGates = true;
            } else if (argument == "--verify" && !shouldVerify) {
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
                throw std::runtime_error("Usage: cuda_state_vector [qubits] [--repetitions M] [--qubit-order ascending|descending] [--verify] [--time-gates] [--baseline-q0] [--q8-pair-swizzle] [--block-size 64|128|256] [--warmup M] [--max-blocks M]");
            }
        }

        check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync),
              "enable blocking synchronization");
        check(cudaSetDevice(0), "select device 0");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0), "query device properties");
        if (properties.maxThreadsPerBlock < static_cast<int>(blockSize) ||
            properties.maxThreadsDim[0] < static_cast<int>(blockSize)) {
            throw std::runtime_error("Device cannot launch requested block size");
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
        const unsigned int deviceLimit = static_cast<unsigned int>(properties.maxGridSize[0]);
        const unsigned int grid = gridFor(pairs, blockSize,
            maxBlocks && maxBlocks < deviceLimit ? maxBlocks : deviceLimit);
        const unsigned int q0Grid = gridFor(amplitudes, blockSize,
            maxBlocks && maxBlocks < deviceLimit ? maxBlocks : deviceLimit);
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
        if (timeGates) {
            resources.gateEvents.resize(qubits + 1, nullptr);
            for (auto& event : resources.gateEvents) {
                check(cudaEventCreateWithFlags(&event, cudaEventBlockingSync),
                      "create gate timing event");
            }
        }
        if (shouldVerify) {
            check(cudaMalloc(reinterpret_cast<void**>(&resources.invalid),
                             sizeof(unsigned int)), "allocate verification flag");
        }
        writeTimingCsvHeader(std::cout);
        for (Index sample = 0; sample < static_cast<Index>(warmup) + repetitions; ++sample) {
            check(cudaMemsetAsync(resources.state, 0, bytes, resources.stream), "zero state vector");
            check(cudaMemcpyAsync(resources.state, &one, sizeof(one),
                                  cudaMemcpyHostToDevice, resources.stream),
                  "initialize amplitude zero");

            check(cudaEventRecord(resources.start, resources.stream), "record start event");
            if (timeGates) {
                check(cudaEventRecord(resources.gateEvents[0], resources.stream),
                      "record first gate boundary");
            }
            for (unsigned int gate = 0; gate < qubits; ++gate) {
                const unsigned int qubit = targetQubit(gate, qubits, qubitOrder);
                if (qubit == 0 && !baselineQubitZero && qubits >= 5) {
                    hadamardQubitZero<<<q0Grid, blockSize, 0, resources.stream>>>(
                        resources.state, amplitudes);
                } else if (useQ8PairSwizzle(q8PairSwizzle, qubit, qubits)) {
                    hadamard<true><<<grid, blockSize, 0, resources.stream>>>(
                        resources.state, pairs, qubit);
                } else {
                    hadamard<<<grid, blockSize, 0, resources.stream>>>(resources.state, pairs, qubit);
                }
                check(cudaGetLastError(), "launch Hadamard kernel");
                if (timeGates) {
                    check(cudaEventRecord(resources.gateEvents[gate + 1], resources.stream),
                          "record gate boundary");
                }
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
                verify<<<grid, blockSize, 0, resources.stream>>>(resources.state, pairs, expected,
                                            resources.invalid);
                check(cudaGetLastError(), "launch verification kernel");
                check(cudaMemcpyAsync(&invalid, resources.invalid, sizeof(invalid),
                                      cudaMemcpyDeviceToHost, resources.stream), "read verification result");
                check(cudaStreamSynchronize(resources.stream), "complete verification");
                if (invalid) throw std::runtime_error("State verification failed");
            }
            if (sample < warmup) continue;
            writeTimingCsvRow(std::cout, sample - warmup + 1, repetitions, milliseconds);
            if (timeGates) {
                for (unsigned int gate = 0; gate < qubits; ++gate) {
                    const unsigned int qubit = targetQubit(gate, qubits, qubitOrder);
                    float gateMilliseconds = 0.0f;
                    check(cudaEventElapsedTime(&gateMilliseconds,
                                              resources.gateEvents[gate],
                                              resources.gateEvents[gate + 1]),
                          "measure gate GPU time");
                    writeTimingCsvRow(std::cout, sample - warmup + 1, repetitions,
                                      gateMilliseconds, static_cast<int>(qubit));
                }
            }
        }
        resources.release();
        if (shouldVerify) std::cerr << "Verification passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
