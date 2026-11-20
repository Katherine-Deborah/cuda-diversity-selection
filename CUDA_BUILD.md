# CUDA Build and Validation

## Compilation on USC CARC

Load CUDA 12.6.3 and compile using:

    module load cuda/12.6.3
    nvcc -O2 --fmad=false -std=c++17 src/kcenter_cuda.cu -o kcenter_cuda

The --fmad=false flag disables fused multiply-add operations to match CPU floating-point calculations.

## Correctness

Tested with 1152-dimensional synthetic data and 100 selected centers on an NVIDIA A40 GPU.

- N=2000: PASS
- N=10000: PASS
- N=20000: PASS

CPU and CUDA produced identical selection sequences with --fmad=false.

## Preliminary Performance

| Points | CPU Time | Original CUDA Time |
|---|---|---|
| 2000 | 0.210392 s | 0.278342 s |
| 10000 | 1.10436 s | 0.26514 s |
| 20000 | 2.2155 s | 0.340549 s |

The original CUDA timings were measured without --fmad=false.

A separate validated CUDA run with --fmad=false took 0.294671 seconds for 20000 points.

## Next Steps

- Validate using the project's actual embedding data.
- Perform additional benchmarks using the validated CUDA configuration.

## Real Embedding Validation

Dataset: `data/pilot_features.npy` (2000 samples, 1152 float32 features).
Converted to a raw float32 binary file for CPU/CUDA input.

Both implementations were compiled with optimization enabled.
CUDA used `--fmad=false` for consistent floating-point behavior.

Results from one real-data run on USC CARC:

| Implementation | Execution time |
|---|---:|
| CPU | 0.200636 s |
| CUDA | 0.232887 s |

Correctness: PASS — CPU and CUDA selected the same 100 centers
in exactly the same order.

At N=2000, GPU overhead exceeds the benefit of parallelization.
The larger synthetic benchmarks demonstrate GPU speedups of
4.25x at N=10000 and 6.39x at N=20000.
