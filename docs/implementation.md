# Implementation and validation

The optional qubit count is an integer from 1 through 59; the default is 33.
N qubits have indices 0 through N−1, so exactly N Hadamard kernels are launched per sample.
Gate order is selected with `--qubit-order ascending|descending` (default
`ascending`). Qubit 0 is the least significant state-index bit; `descending`
applies gates from the most significant qubit N−1 down to 0. This changes gate
execution order, including warmup, without changing state-vector bit numbering.
Gate timing lines follow execution order and retain the actual target labels.
For example:

```sh
./build/cuda_state_vector 8 --qubit-order descending --repetitions 2 --time-gates --verify
```

Expect gate labels 7 through 0 for each sample. Both orders produce the same
final state because Hadamards on distinct qubits commute. Missing or invalid
`--qubit-order` values are rejected.

The circuit is repeated M times (default 128), controlled by `--repetitions M`.
M must be an integer from 1 through the maximum unsigned 32-bit value.
Each sample starts from a freshly initialized all-zero computational basis state.
The upper input bound prevents overflow of the amplitude count and byte count;
actual feasibility is limited by available memory on CUDA device 0. No other GPU
is used, and no cuStateVec dependency is present.

The state uses a Structure of Arrays (SoA) layout: two separate contiguous GPU
allocations, `double* stateReal` and `double* stateImag`, each containing 2^N
FP64 elements. Matching indices form one complex amplitude, so total state
storage remains 16 bytes per amplitude. The memory-capacity check includes both
buffers and the optional verification flag. Both buffers are zeroed before each
sample, then `stateReal[0]` is set to 1.0. Both allocations are released on
success and on error, including failure of the second allocation. Each Hadamard remains an
independent kernel. All launches use scalar X grid and block sizes. The default
block size remains 64; `--block-size 128` and `--block-size 256` allow measurement
without rebuilding. No block size is claimed to be faster without device data.

Each thread processes pairs in a 64-bit grid-stride loop. The grid is the smaller
of ceil(pairs / blockSize) and the device's maximum grid X dimension, optionally
capped further by `--max-blocks M` (positive). Multiplications promote to 64 bits
before evaluating the global thread index and stride. For the supported maximum
of 59 qubits, pairs <= 2^58 and stride <= (2^31 - 1) * 256 on CUDA devices; index
increments cannot overflow. Each pair has one unique residue modulo the stride,
so it is visited exactly once. A grid cap allows exercising repeated iterations
without allocating a huge state vector.

Pair indexing inserts a zero at the target bit using `low = pair & (bit - 1)`
and `first = (pair << 1) - low`. Writing `pair = high * bit + low` shows that
`2 * pair - low = high * 2 * bit + low`, preserving the low bits while shifting
the high bits up by one. At qubit zero this maps pair p to amplitudes 2p
and 2p+1. This source change may reduce address computation, but generated
instructions and measurements are needed to establish any benefit. The loop can
also add overhead; this is not a measured speedup. Target specialization and a
qubit-zero shuffle kernel are deferred pending instruction and transaction data.
FP64 arithmetic is preserved for both real and imaginary components. For qubit
zero, each component buffer has two warp load streams accessing alternating
FP64 elements; higher target bits permit more contiguous accesses within each
stream. The SoA refactor has no measured performance improvement. Cache behavior may
reduce the cost of overlapping memory sectors, so source stride alone does not
establish actual DRAM traffic.

All initialization, Hadamard launches, timing events, and optional verification
use one explicit stream created with `cudaStreamNonBlocking`. Stream ordering
ensures initialization completes before the gates without a device-wide barrier.
CUDA events bracket only the sequence of Hadamard launches in this stream. Allocation, zeroing, initialization, and optional verification precede
or follow this interval. Each reported measurement is one circuit pass, without averaging. Optional
`--warmup M` runs M initialized circuits before reporting samples (default 0);
warmup circuits also undergo verification when requested. Otherwise,
one timing line is printed per sample, including its index and the total count. As with CUDA event timing generally, any GPU idle gaps between
launches within that interval are included. API failures, launch failures, and
asynchronous execution failures are reported to standard error with a nonzero
exit status. The program rejects insufficient free memory before allocation;
allocation can still fail if available memory changes after the check.

With `--time-gates`, each reported sample additionally prints one GPU time in
milliseconds per Hadamard gate, labeled by its zero-based target qubit. Timing
is disabled by default. An event at each gate boundary is recorded in the same
stream; elapsed times are read only after the circuit stop event completes,
without synchronizing between gates. Events are allocated once and reused for
all samples. Warmup gate times are not printed. Initialization and verification
remain outside these measurements. Event intervals can include launch gaps;
additional event recording can perturb both gate and circuit times, so compare
performance using the same timing mode.

For a small GPU smoke test, run:

```sh
./build/cuda_state_vector 8 --repetitions 2 --warmup 1 --time-gates --verify
```

Expect two sample totals, each followed by eight gate timings for qubits 0–7,
and successful verification. Without `--time-gates`, expect only sample totals.

Host waits use blocking synchronization: `cudaDeviceScheduleBlockingSync` is set
before device initialization for stream synchronization, and both timing events
are created with `cudaEventBlockingSync`. Event timing remains enabled.
The stream's `cudaStreamNonBlocking` flag controls implicit synchronization with
the default stream independently of how the host waits.
The stop event is synchronized before reading each timing result. Verification
uses an asynchronous result copy followed by stream synchronization before the
host reads the flag. Host copy buffers remain alive until stream work completes;
these pageable buffers do not guarantee transfer overlap. Cleanup waits for the
stream before releasing allocations and destroys the stream on both success and
error paths.

With `--verify`, a separate kernel checks every final amplitude after each sample against
`2^(-N/2) + 0i`, using a relative tolerance of `1e-12`, and rejects non-finite
values. This check and its memory allocation are outside the measured interval.

On a CUDA-equipped machine, build as shown in the README and smoke-test:

```sh
./build/cuda_state_vector 1 --verify
./build/cuda_state_vector 8 --repetitions 2 --verify
./build/cuda_state_vector 20 --verify
./build/cuda_state_vector 33 --verify
```

The last command requires more than 128 GiB of available GPU memory including
runtime overhead. On a smaller GPU it must fail with an insufficient-memory
error. Invalid inputs such as `0`, `60`, and `abc` must also fail. `--repetitions 0`, `--repetitions -1`, `--repetitions abc`, a missing value,
and values above 4294967295 must also fail. Default runs must print 128 timing
lines; `--repetitions 1` and `--repetitions 2` must print one and two respectively,
including successful verification when requested. For a device
memory error check, run `compute-sanitizer --tool memcheck` on the 8-qubit case.


## Optimization validation status

This environment has no CUDA compiler or visible NVIDIA tooling, and contains no
`custatevec-study` checkout. No CUDA compilation, GPU correctness, instruction
inspection, or performance comparison was possible. Performance hypotheses above
are unverified; the 64-thread default is retained deliberately.

`make test` needs only a C++17 host compiler and checks exhaustive small-vector
pair coverage, capped grid-stride coverage for all three block sizes, and index
round trips through the 59-qubit boundary, including indices above 2^32.
It does not allocate a large state vector or substitute for GPU validation.
The host test passed here with warnings treated as errors. An additional UBSan
build could not link because the environment lacks the UBSan runtime library.
`make test-cuda` uses the CUDA configuration and compares each individual gate
against an independent CPU reference with nonzero real and imaginary inputs,
for every target of 1-, 8-, and 12-qubit states, all three block sizes, and grid
caps of 1, 3, and the normal device limit. Run it under compute-sanitizer too.
Actual large allocations still require hardware with sufficient free memory.

## Reproducible GPU comparison

Build the original revision `a5d68c8` separately (for example from a git archive,
without switching the current branch) using the same toolkit, architecture flags,
and optimization flags. Run on the same idle GPU with the same qubit count,
ascending Hadamards, FP64 complex storage, one device, and fusion disabled in
all implementations. Match initialization and timing boundaries in
`custatevec-study`; do not compare timings that include allocation with kernel
sequence timings. Record GPU model, clocks, driver, toolkit and compiler flags.

For example, on a GPU that fits 24 qubits:

```sh
make test
make test-cuda
for block in 64 128 256; do
  ./build/cuda_state_vector 24 --block-size "$block" --warmup 10 --repetitions 128 --verify
done
./build/cuda_state_vector 12 --max-blocks 1 --repetitions 2 --verify
compute-sanitizer --tool memcheck ./build/test_kernels
```

Collect matching unprofiled timing samples from all implementations and report
median and spread. For the original executable, which has no explicit warmup,
run 138 samples and discard the first 10. Verification is outside the timed
interval but can affect subsequent cache/thermal conditions; use the same
verification policy and repeat without verification for performance comparisons.
The existing CUDA events report the full gate sequence including launch gaps.
Use Nsight Compute to collect individual target-qubit kernel durations, DRAM
read/write bytes, memory sector utilization, occupancy and register counts in
separate profiling runs. Report effective bandwidth as 32 * amplitudes / seconds
(one 16-byte read and write per amplitude), explicitly distinguishing this
logical bandwidth from measured DRAM traffic. Inspect SASS with `cuobjdump
--dump-sass build/cuda_state_vector` and build with `NVCCFLAGS="-O3 -lineinfo
-Xptxas=-v"` to compare shift/address instructions and register use against the
original, especially for qubit zero. Profiler replay timings must not replace
unprofiled circuit measurements. Choose a new default or specialization only
when these results support it.
