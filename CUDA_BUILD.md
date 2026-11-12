# CUDA Build and Validation

## Compilation on USC CARC

Load CUDA and compile:

```bash
module load cuda/12.6.3
nvcc -O2 --fmad=false -std=c++17 src/kcenter_cuda.cu -o kcenter_cuda

cat > CUDA_BUILD.md <<'EOF'
# CUDA Build and Validation

## Build
On USC CARC, load CUDA 12.6.3 and compile with:

    nvcc -O2 --fmad=false -std=c++17 src/kcenter_cuda.cu -o kcenter_cuda

The --fmad=false flag disables fused multiply-add operations to match CPU floating-point calculations.

## Correctness
Tested with 1152-dimensional synthetic data and 100 selected centers on an NVIDIA A40 GPU.

- N=2000: PASS
- N=10000: PASS
- N=20000: PASS

CPU and CUDA produced identical selection sequences with --fmad=false.

## Performance
- N=2000: CPU 0.210392s, original CUDA 0.278342s
- N=10000: CPU 1.10436s, original CUDA 0.26514s
- N=20000: CPU 2.2155s, original CUDA 0.340549s

Validated CUDA run for N=20000: 0.294671s.

Original CUDA timings were measured without --fmad=false.
