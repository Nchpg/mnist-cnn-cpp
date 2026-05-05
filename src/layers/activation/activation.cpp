#include "layers/activation/activation.hpp"

#include <cmath>

scalar_t Activation::relu(scalar_t x)
{
    return x > 0.0f ? x : 0.0f;
}

scalar_t Activation::relu_deriv(scalar_t x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}

void Activation::softmax(const Matrix &logits, Matrix &probs)
{
    size_t num_rows = logits.rows();
    size_t cols = logits.cols();
    if (probs.rows() != num_rows || probs.cols() != cols)
    {
        throw std::invalid_argument("Dimension mismatch in softmax");
    }

    for (size_t j = 0; j < cols; ++j)
    {
        scalar_t max_val = logits(0, j);
        for (size_t i = 1; i < num_rows; ++i)
        {
            if (logits(i, j) > max_val)
                max_val = logits(i, j);
        }

        scalar_t sum = 0.0f;
        for (size_t i = 0; i < num_rows; ++i)
        {
            probs(i, j) = std::exp(logits(i, j) - max_val);
            sum += probs(i, j);
        }
        scalar_t inv_sum = 1.0f / sum;
        for (size_t i = 0; i < num_rows; ++i)
        {
            probs(i, j) *= inv_sum;
        }
    }
}