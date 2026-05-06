#include "loss/mse_loss.hpp"

#include <stdexcept>

scalar_t MSELoss::forward(const Tensor& predictions, const std::vector<size_t>& targets)
{
    size_t batch_size = predictions.shape()[0];
    size_t num_classes = predictions.shape()[1];
    if (targets.size() != batch_size)
        throw std::invalid_argument("Targets size must match batch size");

    scalar_t total_loss = 0.0f;
    for (size_t b = 0; b < batch_size; ++b)
    {
        size_t t = targets[b];
        if (t >= num_classes)
            throw std::out_of_range("Target index out of range");

        scalar_t diff = predictions(b, t) - 1.0f;
        total_loss += diff * diff;

        for (size_t c = 0; c < num_classes; ++c)
        {
            if (c == t)
                continue;
            scalar_t diff2 = predictions(b, c) - 0.0f;
            total_loss += diff2 * diff2;
        }
    }
    return total_loss / static_cast<scalar_t>(num_classes * batch_size);
}

void MSELoss::backward(const Tensor& predictions, const std::vector<size_t>& targets, Tensor& grad_output)
{
    size_t batch_size = predictions.shape()[0];
    size_t num_classes = predictions.shape()[1];
    if (targets.size() != batch_size)
        throw std::invalid_argument("Targets size must match batch size");

    if (grad_output.rank() != 2 || grad_output.shape()[0] != batch_size || grad_output.shape()[1] != num_classes)
    {
        grad_output.reshape(Shape({ batch_size, num_classes }));
    }

    scalar_t coeff = 2.0f / static_cast<scalar_t>(num_classes * batch_size);
    for (size_t b = 0; b < batch_size; ++b)
    {
        size_t t = targets[b];
        for (size_t c = 0; c < num_classes; ++c)
        {
            scalar_t target_val = (c == t) ? 1.0f : 0.0f;
            grad_output(b, c) = coeff * (predictions(b, c) - target_val);
        }
    }
}