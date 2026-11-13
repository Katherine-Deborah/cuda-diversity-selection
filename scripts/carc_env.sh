# Source after activating ee451-pilot. Paths do not touch the original dataset.
export HF_HOME="/scratch1/${USER}/ee451-cuda-diversity/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export EE451_PILOT_ROOT="/scratch1/${USER}/ee451-katherine-pilot"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$EE451_PILOT_ROOT"
