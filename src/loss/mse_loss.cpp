#include "loss/mse_loss.hpp"

#include <stdexcept>
#include <vector>

scalar_t MSELoss::forward(const Matrix &predictions,
                          const std::vector<size_t> &targets)
{
    size_t num_cols = predictions.cols();
    size_t num_rows = predictions.rows();
    if (targets.size() != num_cols)
    {
        throw std::invalid_argument("Targets size must match batch size");
    }

    scalar_t total_loss = 0.0f;
    for (size_t j = 0; j < num_cols; ++j)
    {
        size_t t = targets[j];
        if (t >= num_rows)
        {
            throw std::out_of_range("Target index out of range");
        }
        scalar_t diff = predictions(t, j) - 1.0f;
        total_loss += diff * diff;

        for (size_t i = 0; i < num_rows; ++i)
        {
            if (i == t)
                continue;
            scalar_t diff2 = predictions(i, j) - 0.0f;
            total_loss += diff2 * diff2;
        }
    }

    return total_loss / static_cast<scalar_t>(num_rows * num_cols);
}

void MSELoss::backward(const Matrix &predictions,
                       const std::vector<size_t> &targets, Matrix &grad_output)
{
    size_t num_rows = predictions.rows();
    size_t num_cols = predictions.cols();
    if (targets.size() != num_cols)
    {
        throw std::invalid_argument("Targets size must match batch size");
    }
    if (grad_output.rows() != num_rows || grad_output.cols() != num_cols)
    {
        grad_output.reshape(num_rows, num_cols);
    }

    scalar_t coeff = 2.0f / static_cast<scalar_t>(num_rows * num_cols);
    for (size_t j = 0; j < num_cols; ++j)
    {
        size_t t = targets[j];
        if (t >= num_rows)
        {
            throw std::out_of_range("Target index out of range");
        }
        for (size_t i = 0; i < num_rows; ++i)
        {
            scalar_t target_val = (i == t) ? 1.0f : 0.0f;
            grad_output(i, j) = coeff * (predictions(i, j) - target_val);
        }
    }
}