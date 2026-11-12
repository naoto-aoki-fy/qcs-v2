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

Each complex amplitude is a `double2` (16 bytes). The initial state is
amplitude 1 at index zero and 0 everywhere else. Each Hadamard remains an
independent kernel. All launches use scalar X grid and block sizes. The default
block size remains 64; `--block-size 128` and `--block-size 256` allow measurement
without rebuilding. No block size is claimed to be faster without device data.

For targets above zero (and the optional q=0 baseline), each thread processes
pairs in a 64-bit grid-stride loop. The grid is the smaller
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
also add overhead; this is not a measured speedup. The q=0 shuffle specialization described below is available for comparison.
The double2 layout and FP64 arithmetic are preserved. For qubit zero, each of
the two warp load streams accesses alternating complex elements; higher target
bits permit more contiguous accesses within each stream. Cache behavior may
reduce the cost of overlapping memory sectors, so source stride alone does not
establish actual DRAM traffic.

All initialization, Hadamard launches, timing events, and optional verification
use one explicit stream created with `cudaStreamNonBlocking`. Stream ordering
ensures initialization completes before the gates without a device-wide barrier.
CUDA events bracket only the sequence of Hadamard launches in this stream. Allocation, zeroing, initialization, and optional verification precede
or follow this interval. Each reported measurement is one circuit pass, without averaging. Optional
`--warmup M` runs M initialized circuits before reporting samples (default 0);
warmup circuits also undergo verification when requested. Otherwise,
one CSV total row is printed per sample, including its index and the total count. As with CUDA event timing generally, any GPU idle gaps between
launches within that interval are included. API failures, launch failures, and
asynchronous execution failures are reported to standard error with a nonzero
exit status. The program rejects insufficient free memory before allocation;
allocation can still fail if available memory changes after the check.

With `--time-gates`, each reported sample additionally prints one GPU time in
milliseconds per Hadamard gate, labeled by its zero-based target qubit. Per-gate timing
is disabled by default. An event at each gate boundary is recorded in the same
stream; elapsed times are read only after the circuit stop event completes,
without synchronizing between gates. Events are allocated once and reused for
all samples. Warmup gate times are not printed. Initialization and verification
remain outside these measurements. Event intervals can include launch gaps;
additional event recording can perturb both gate and circuit times, so compare
performance using the same timing mode.

Timing output is CSV on standard output, with a single header:

```csv
type,sample,repetitions,qubit,milliseconds
total,1,2,,0.123456
gate,1,2,0,0.012345
```

`total` rows contain the Hadamard kernel sequence time and an empty qubit
field. With `--time-gates`, `gate` rows follow each total in execution order,
with the zero-based target qubit. Sample indices start at 1 after warmup.
Times use milliseconds with six decimal places; displayed precision does not
imply measurement accuracy. Verification success and errors go to standard
error so they do not contaminate the CSV. Redirect standard output to save it:

```sh
./build/cuda_state_vector 8 --repetitions 2 --time-gates --verify > timings.csv
```

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
for every target of 1-, 2-, 3-, 4-, 5-, 6-, 8-, and 12-qubit states, all three block sizes, and grid
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


## Qubit-zero warp shuffle

By default, q=0 with N>=5 uses `hadamardQubitZero`: each thread loads and stores one
contiguous `double2` amplitude. Two FP64 `__shfl_xor_sync` operations with lane
mask 1 exchange real and imaginary components. Even lanes write `(a+b)*scale`;
odd lanes write `(a-b)*scale`, with the same FP64 scale as the pair kernel.
Targets q>=1 and states with N<5 use the pair kernel and its launch dimensions.

Index and stride arithmetic remain 64-bit. The q=0 grid is sized by amplitudes
rather than pairs, subject to the same grid cap. Supported blocks are multiples
of 32. For N>=5, the amplitude count is also a multiple of 32. Each warp
iterates uniformly with all 32 lanes in bounds, so both shuffles use the fixed
participation mask `0xffffffffu` without a ballot or per-lane bounds branch. Both shuffle operations complete
before writes, so each pair has loaded its inputs before either output can
overwrite them. Each thread owns a unique amplitude; no additional buffer or
cross-warp synchronization is needed.

`--baseline-q0` selects the original pair kernel for q=0 for A/B measurements.
Use identical options on the same idle GPU, for example (32 qubits need 64 GiB
for the state vector plus runtime overhead):

```sh
make test
make test-cuda
./build/cuda_state_vector 32 --warmup 10 --repetitions 128 --time-gates --verify --baseline-q0
./build/cuda_state_vector 32 --warmup 10 --repetitions 128 --time-gates --verify
```

Alternate baseline and shuffle runs and compare medians and spread for q=0
and q>=1; repeat with descending order and each supported block size. Tests
compare both q=0 kernels directly as well as against the CPU reference, covering
the N<5 pair fallback, the N=5 full-warp boundary, and repeated grid-stride
iterations. Run the CUDA tests under
compute-sanitizer on a GPU as well.

No CUDA compiler or NVIDIA GPU tooling is available in the implementation
environment. CUDA compilation, GPU correctness, and performance have not been
measured; the specialization is pending GPU validation and no speedup is claimed.
The baseline switch remains available if measurements show a regression.

## Experimental q=8 pair permutation

`--q8-pair-swizzle` enables an experimental instantiation of the pair kernel
only for target q=8 with N>=10. The default remains the original pair mapping;
N=9 and all other targets retain their existing paths. This switch combines
with all existing options, including gate order and `--baseline-q0`.

Before inserting the target bit, the experiment swaps bits 5 and 8 of the
logical pair index. The loop counter and grid stride remain unchanged. Swapping
twice restores the original index, proving bijectivity on the power-of-two
pair domain when both bits exist (N>=10). Original grid-stride coverage thus
implies each mapped pair, and each of its two distinct amplitudes, is visited
exactly once. There is no overlap between writers and no extra synchronization,
state storage, or change to gate order or FP64 arithmetic.

The lower five pair bits are preserved, so consecutive lanes still access
consecutive amplitudes in each of the two streams. The baseline q=8 accesses
are already coalesced. This experiment changes warp/block ownership of pairs,
which may alter cache locality, memory partition traffic, and scheduling.
Additional integer instructions or register pressure may offset any benefit;
there is no measured performance improvement.

`make test` exhaustively checks the permutation, inverse, lane bits, bounds,
and unique pair/amplitude ownership for N=10..16, blocks 64/128/256, and grid
caps 1/2/3/uncapped. It also checks dispatch eligibility, including N=9.
`make test-cuda` includes N=9/10/12, compares the experimental path directly
with the baseline and an independent complex-valued CPU reference, and checks
full circuits in both orders for each block size and caps 1/3/uncapped.
Host tests passed in the implementation environment. CUDA compilation, GPU
correctness, sanitizer checks, and performance remain unverified because no
CUDA compiler or NVIDIA GPU tooling is available.

On a configured GPU, run `make test-cuda` and
`compute-sanitizer --tool memcheck ./build/test_kernels`. Also test the actual
CLI and verification kernel for N=9 (fallback), 10, and 12 with each block size,
both orders, and grid caps 1 and 3, for example:

```sh
./build/cuda_state_vector 10 --block-size 128 --max-blocks 1 --warmup 1 \
  --repetitions 2 --qubit-order descending --time-gates --verify --q8-pair-swizzle
```

For performance, use the same executable and configuration for both paths:

```sh
./build/cuda_state_vector 32 --block-size 64 --warmup 10 --repetitions 30 \
  --time-gates > baseline.csv
./build/cuda_state_vector 32 --block-size 64 --warmup 10 --repetitions 30 \
  --time-gates --q8-pair-swizzle > swizzle.csv
```

Alternate baseline and experimental runs on the same idle GPU. Keep N, block
size, grid cap, gate order, verification policy, compiler flags, GPU clocks,
and timing mode identical. Record hardware, driver, and toolkit versions.
For each CSV report the median, interquartile range (25th–75th percentiles),
and minimum/maximum separately for q=7, q=8, q=9 and total circuit time.
Repeat with each supported block size and relevant grid caps. A 32-qubit
state needs 64 GiB plus runtime overhead; use a smaller N>=10 if necessary.
