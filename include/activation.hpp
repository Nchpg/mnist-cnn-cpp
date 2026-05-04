#pragma once

#include "matrix.hpp"
#include <cmath>
#include <stdexcept>
#include <functional>
#include <vector>

namespace Activation {
    using ElementFunc = scalar_t(*)(scalar_t);

    inline scalar_t relu(scalar_t x) { return x > 0.0f ? x : 0.0f; }
    inline scalar_t relu_deriv(scalar_t x) { return x > 0.0f ? 1.0f : 0.0f; }

    inline void softmax(const Matrix& logits, Matrix& probs) {
        size_t num_rows = logits.rows(); 
        size_t cols = logits.cols();
        if (probs.rows() != num_rows || probs.cols() != cols) {
            throw std::invalid_argument("Dimension mismatch in softmax");
        }

        for (size_t j = 0; j < cols; ++j) {
            scalar_t max_val = logits(0, j);
            for (size_t i = 1; i < num_rows; ++i) {
                if (logits(i, j) > max_val) max_val = logits(i, j);
            }

            scalar_t sum = 0.0f;
            for (size_t i = 0; i < num_rows; ++i) {
                probs(i, j) = std::exp(logits(i, j) - max_val);
                sum += probs(i, j);
            }
            scalar_t inv_sum = 1.0f / sum;
            for (size_t i = 0; i < num_rows; ++i) {
                probs(i, j) *= inv_sum;
            }
        }
    }

    inline scalar_t cross_entropy_loss_stable(const Matrix& probs, const std::vector<size_t>& targets) {
        size_t num_cols = probs.cols(); 
        if (targets.size() != num_cols) {
            throw std::invalid_argument("Targets size must match batch size");
        }

        scalar_t total_loss = 0.0f;
        const scalar_t epsilon = 1e-7f; 
        for (size_t j = 0; j < num_cols; ++j) {
            total_loss += -std::log(std::max(probs(targets[j], j), epsilon));
        }
        return total_loss / static_cast<scalar_t>(num_cols);
    }

    inline std::vector<size_t> argmax(const Matrix& m) {
        std::vector<size_t> results(m.cols());
        for (size_t j = 0; j < m.cols(); ++j) {
            size_t best = 0;
            for (size_t i = 1; i < m.rows(); ++i) {
                if (m(i, j) > m(best, j)) best = i;
            }
            results[j] = best;
        }
        return results;
    }
}