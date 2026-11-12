#include <iostream>
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <cstddef>

using namespace std;

// Squared Euclidean distance between two feature vectors.
float squared_distance(const float* a, const float* b, size_t dim) {
    float distance = 0.0f;

    for (size_t j = 0; j < dim; ++j) {
        float diff = a[j] - b[j];
        distance += diff * diff;
    }

    return distance;
}

// Greedy k-center selection using the CPU.
vector<size_t> kcenter_cpu(
    const vector<float>& X,
    size_t N,
    size_t dim,
    size_t k,
    size_t start = 0
) {
    if (N == 0 || dim == 0 || k == 0 || k > N || start >= N) {
        throw invalid_argument("Invalid N, dim, k, or start index");
    }

    if (X.size() != N * dim) {
        throw invalid_argument("Feature matrix size does not match N * dim");
    }

    vector<size_t> selected;
    selected.reserve(k);

    vector<float> nearest_distance(N, numeric_limits<float>::infinity());
    vector<bool> is_selected(N, false);

    selected.push_back(start);
    is_selected[start] = true;

    // Initialize distances from the first selected center.
    for (size_t i = 0; i < N; ++i) {
        nearest_distance[i] = squared_distance(
            &X[i * dim],
            &X[start * dim],
            dim
        );
    }

    nearest_distance[start] = -1.0f;

    // Select the remaining k - 1 centers.
    for (size_t iteration = 1; iteration < k; ++iteration) {

        size_t farthest_index = N;
        float max_distance = -1.0f;

        // Find the farthest unselected point.
        for (size_t i = 0; i < N; ++i) {
            if (!is_selected[i] && nearest_distance[i] > max_distance) {
                max_distance = nearest_distance[i];
                farthest_index = i;
            }
        }

        if (farthest_index == N) {
            throw runtime_error("No unselected center found");
        }

        selected.push_back(farthest_index);
        is_selected[farthest_index] = true;

        // Update distances to the nearest selected center.
        for (size_t i = 0; i < N; ++i) {
            if (is_selected[i]) {
                nearest_distance[i] = -1.0f;
                continue;
            }

            float distance = squared_distance(
                &X[i * dim],
                &X[farthest_index * dim],
                dim
            );

            nearest_distance[i] = min(nearest_distance[i], distance);
        }
    }

    return selected;
}

int main() {
    // Small example for correctness testing.
    const size_t N = 5;
    const size_t dim = 2;
    const size_t k = 3;

    vector<float> X = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        5.0f, 5.0f,
        6.0f, 5.0f
    };

    try {
        vector<size_t> selected = kcenter_cpu(X, N, dim, k);

        cout << "Selected indices: ";

        for (size_t index : selected) {
            cout << index << " ";
        }

        cout << endl;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}