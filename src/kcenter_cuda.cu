#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cfloat>

// Check CUDA calls so errors are easier to find.
void check_cuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: "
                  << cudaGetErrorString(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Each thread updates the distance for one point.
// We keep the distance to the closest center selected so far.
__global__ void update_distances(
    const float* X,
    const float* center,
    float* distances,
    int N,
    int dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) {
        return;
    }

    float dist = 0.0f;

    for (int j = 0; j < dim; ++j) {
        float diff = X[i * dim + j] - center[j];
        dist += diff * diff;
    }

    distances[i] = fminf(distances[i], dist);
}

// Find the farthest unselected point in each block.
// If distances are equal, use the smaller index.
__global__ void reduce_max(
    const float* distances,
    const int* selected,
    float* block_max,
    int* block_index,
    int N
) {
    extern __shared__ unsigned char shared_memory[];

    float* shared_dist = reinterpret_cast<float*>(shared_memory);
    int* shared_idx = reinterpret_cast<int*>(&shared_dist[blockDim.x]);

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float value = -1.0f;
    int index = N;

    if (i < N && selected[i] == 0) {
        value = distances[i];
        index = i;
    }

    shared_dist[tid] = value;
    shared_idx[tid] = index;

    __syncthreads();

    // Reduce the distances within the block.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            float other_value = shared_dist[tid + stride];
            int other_index = shared_idx[tid + stride];

            if (other_value > shared_dist[tid] ||
                (other_value == shared_dist[tid] &&
                 other_index < shared_idx[tid])) {

                shared_dist[tid] = other_value;
                shared_idx[tid] = other_index;
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = shared_dist[0];
        block_index[blockIdx.x] = shared_idx[0];
    }
}

// Greedy k-center using CUDA for distance updates and block reduction.
std::vector<int> kcenter_cuda(
    const std::vector<float>& X,
    int N,
    int dim,
    int k,
    int start = 0
) {
    if (N <= 0 || dim <= 0 || k <= 0 || k > N ||
        start < 0 || start >= N ||
        X.size() != static_cast<size_t>(N) * dim) {
        throw std::invalid_argument("Invalid k-center inputs");
    }

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    float* d_X = nullptr;
    float* d_distances = nullptr;
    float* d_center = nullptr;
    float* d_block_max = nullptr;

    int* d_selected = nullptr;
    int* d_block_index = nullptr;

    // Allocate GPU memory.
    check_cuda(cudaMalloc((void**)&d_X, X.size() * sizeof(float)));
    check_cuda(cudaMalloc((void**)&d_distances, N * sizeof(float)));
    check_cuda(cudaMalloc((void**)&d_center, dim * sizeof(float)));
    check_cuda(cudaMalloc((void**)&d_selected, N * sizeof(int)));
    check_cuda(cudaMalloc((void**)&d_block_max, blocks * sizeof(float)));
    check_cuda(cudaMalloc((void**)&d_block_index, blocks * sizeof(int)));

    // Copy the feature matrix to the GPU.
    check_cuda(cudaMemcpy(
        d_X, X.data(), X.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    std::vector<int> selected_flags(N, 0);
    std::vector<int> selected_indices;
    selected_indices.reserve(k);

    std::vector<float> distances(N, FLT_MAX);
    std::vector<float> block_max(blocks);
    std::vector<int> block_index(blocks);

    check_cuda(cudaMemcpy(
        d_distances, distances.data(), N * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    int current = start;

    for (int iteration = 0; iteration < k; ++iteration) {
        selected_indices.push_back(current);
        selected_flags[current] = 1;

        // Copy selected flags and the new center to the GPU.
        check_cuda(cudaMemcpy(
            d_selected, selected_flags.data(), N * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        check_cuda(cudaMemcpy(
            d_center, X.data() + static_cast<size_t>(current) * dim,
            dim * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        // Update nearest-center distances.
        update_distances<<<blocks, threads>>>(
            d_X, d_center, d_distances, N, dim
        );
        check_cuda(cudaGetLastError());

        if (iteration == k - 1) {
            break;
        }

        // Find the farthest unselected point in each block.
        size_t shared_bytes =
            threads * (sizeof(float) + sizeof(int));

        reduce_max<<<blocks, threads, shared_bytes>>>(
            d_distances,
            d_selected,
            d_block_max,
            d_block_index,
            N
        );
        check_cuda(cudaGetLastError());

        // Copy block results back to the CPU.
        check_cuda(cudaMemcpy(
            block_max.data(), d_block_max,
            blocks * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        check_cuda(cudaMemcpy(
            block_index.data(), d_block_index,
            blocks * sizeof(int),
            cudaMemcpyDeviceToHost
        ));

        // Combine the block results and break ties by index.
        float best_distance = -1.0f;
        int best_index = N;

        for (int b = 0; b < blocks; ++b) {
            if (block_max[b] > best_distance ||
                (block_max[b] == best_distance &&
                 block_index[b] < best_index)) {

                best_distance = block_max[b];
                best_index = block_index[b];
            }
        }

        if (best_index == N) {
            throw std::runtime_error("No unselected center found");
        }

        current = best_index;
    }

    // Free GPU memory.
    check_cuda(cudaFree(d_X));
    check_cuda(cudaFree(d_distances));
    check_cuda(cudaFree(d_center));
    check_cuda(cudaFree(d_selected));
    check_cuda(cudaFree(d_block_max));
    check_cuda(cudaFree(d_block_index));

    return selected_indices;
}

int main() {
    // Small dataset to check against the Python reference.
    const int N = 5;
    const int dim = 2;
    const int k = 3;

    std::vector<float> X = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        5.0f, 5.0f,
        6.0f, 5.0f
    };

    std::vector<int> selected = kcenter_cuda(X, N, dim, k);

    std::cout << "CUDA selected indices: ";

    for (int index : selected) {
        std::cout << index << " ";
    }

    std::cout << std::endl;

    return 0;
}