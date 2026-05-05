#include "loss/cross_entropy_loss.hpp"

#include <stdexcept>

scalar_t CrossEntropyLoss::forward(const Matrix &predictions,
                                   const std::vector<size_t> &targets)
{
    size_t num_classes = predictions.rows();
    size_t num_cols = predictions.cols();
    if (targets.size() != num_cols)
    {
        throw std::invalid_argument("Targets size must match batch size");
    }

    const scalar_t sum_tol = 0.01f;
    for (size_t j = 0; j < num_cols; ++j)
    {
        scalar_t col_sum = 0.0f;
        for (size_t i = 0; i < num_classes; ++i)
        {
            scalar_t val = predictions(i, j);
            if (val < 0.0f)
            {
                throw std::runtime_error(
                    "Predictions must be positive (require Softmax output)");
            }
            col_sum += val;
        }

        if (std::abs(col_sum - 1.0f) > sum_tol)
        {
            throw std::runtime_error(
                "Predictions must sum to 1 (require Softmax output). "
                "Got sum " + std::to_string(col_sum));
        }
    }

    scalar_t total_loss = 0.0f;
    const scalar_t epsilon = 1e-7f;
    for (size_t j = 0; j < num_cols; ++j)
    {
        total_loss += -std::log(std::max(predictions(targets[j], j), epsilon));
    }
    return total_loss / static_cast<scalar_t>(num_cols);
}

void CrossEntropyLoss::backward(const Matrix &predictions,
                                const std::vector<size_t> &targets,
                                Matrix &grad_output)
{
    size_t num_classes = predictions.rows();
    size_t batch_size = predictions.cols();

    if (grad_output.rows() != num_classes || grad_output.cols() != batch_size)
    {
        grad_output.reshape(num_classes, batch_size);
    }

    scalar_t inv_batch_size = 1.0f / static_cast<scalar_t>(batch_size);

#pragma omp parallel for collapse(2) if (num_classes * batch_size > 1000)
    for (size_t i = 0; i < num_classes; ++i)
    {
        for (size_t b = 0; b < batch_size; ++b)
        {
            scalar_t target = (i == targets[b] ? 1.0f : 0.0f);

            if (target > 0.0f)
            {
                grad_output(i, b) =
                    -inv_batch_size / std::max(predictions(i, b), 1e-7f);
            }
            else
            {
                grad_output(i, b) = 0.0f;
            }
        }
    }
}

bool CrossEntropyLoss::supports_fusion_with(const std::string &layer_type) const
{
    return layer_type == "Softmax";
}

void CrossEntropyLoss::backward_fused(const std::string &layer_type,
                                      const Matrix &probs,
                                      const std::vector<size_t> &targets,
                                      Matrix &grad_logits)
{
    if (layer_type != "Softmax")
    {
        throw std::invalid_argument("CrossEntropyLoss cannot fuse with "
                                    + layer_type);
    }

    size_t num_classes = probs.rows();
    size_t batch_size = probs.cols();

    if (grad_logits.rows() != num_classes || grad_logits.cols() != batch_size)
    {
        grad_logits.reshape(num_classes, batch_size);
    }

    scalar_t inv_batch_size = 1.0f / static_cast<scalar_t>(batch_size);

#pragma omp parallel for collapse(2) if (num_classes * batch_size > 1000)
    for (size_t i = 0; i < num_classes; ++i)
    {
        for (size_t b = 0; b < batch_size; ++b)
        {
            grad_logits(i, b) = probs(i, b) * inv_batch_size;
        }
    }

    for (size_t b = 0; b < batch_size; ++b)
    {
        grad_logits(targets[b], b) -= inv_batch_size;
    }
}