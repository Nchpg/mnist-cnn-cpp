#include "loss/cross_entropy_loss.hpp"

#include <cmath>
#include <stdexcept>

scalar_t CrossEntropyLoss::forward(const Tensor &predictions,
                                   const std::vector<size_t> &targets)
{
    size_t batch_size = predictions.shape()[0];
    size_t num_classes = predictions.shape()[1];
    if (targets.size() != batch_size)
        throw std::invalid_argument("Targets size must match batch size");

    const scalar_t sum_tol = 0.01f;
    for (size_t b = 0; b < batch_size; ++b)
    {
        scalar_t row_sum = 0.0f;
        for (size_t c = 0; c < num_classes; ++c)
        {
            scalar_t val = predictions(b, c);
            if (val < 0.0f)
                throw std::runtime_error(
                    "Predictions must be positive (require Softmax output)");
            row_sum += val;
        }
        if (std::abs(row_sum - 1.0f) > sum_tol)
        {
            throw std::runtime_error(
                "Predictions must sum to 1 (require Softmax). Got sum "
                + std::to_string(row_sum));
        }
    }

    scalar_t total_loss = 0.0f;
    const scalar_t epsilon = 1e-7f;
    for (size_t b = 0; b < batch_size; ++b)
    {
        total_loss += -std::log(std::max(predictions(b, targets[b]), epsilon));
    }
    return total_loss / static_cast<scalar_t>(batch_size);
}

void CrossEntropyLoss::backward(const Tensor &predictions,
                                const std::vector<size_t> &targets,
                                Tensor &grad_output)
{
    size_t batch_size = predictions.shape()[0];
    size_t num_classes = predictions.shape()[1];

    if (grad_output.rank() != 2 || grad_output.shape()[0] != batch_size
        || grad_output.shape()[1] != num_classes)
    {
        grad_output.reshape(Shape({ batch_size, num_classes }));
    }

    scalar_t inv_batch_size = 1.0f / static_cast<scalar_t>(batch_size);

#pragma omp parallel for collapse(2) if (num_classes * batch_size > 1000)
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t c = 0; c < num_classes; ++c)
        {
            scalar_t target = (c == targets[b] ? 1.0f : 0.0f);
            if (target > 0.0f)
            {
                grad_output(b, c) =
                    -inv_batch_size / std::max(predictions(b, c), 1e-7f);
            }
            else
            {
                grad_output(b, c) = 0.0f;
            }
        }
    }
}

bool CrossEntropyLoss::supports_fusion_with(const std::string &layer_type) const
{
    return layer_type == "Softmax";
}

void CrossEntropyLoss::backward_fused(const std::string &layer_type,
                                      const Tensor &probs,
                                      const std::vector<size_t> &targets,
                                      Tensor &grad_logits)
{
    if (layer_type != "Softmax")
        throw std::invalid_argument("CrossEntropyLoss cannot fuse with "
                                    + layer_type);

    size_t batch_size = probs.shape()[0];
    size_t num_classes = probs.shape()[1];

    if (grad_logits.rank() != 2 || grad_logits.shape()[0] != batch_size
        || grad_logits.shape()[1] != num_classes)
    {
        grad_logits.reshape(Shape({ batch_size, num_classes }));
    }

    scalar_t inv_batch_size = 1.0f / static_cast<scalar_t>(batch_size);

#pragma omp parallel for collapse(2) if (num_classes * batch_size > 1000)
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t c = 0; c < num_classes; ++c)
        {
            grad_logits(b, c) = probs(b, c) * inv_batch_size;
        }
    }

    for (size_t b = 0; b < batch_size; ++b)
    {
        grad_logits(b, targets[b]) -= inv_batch_size;
    }
}