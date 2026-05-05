#include "loss/cross_entropy_loss.hpp"
#include <stdexcept>

scalar_t CrossEntropyLoss::forward(const Matrix &predictions,
                                const std::vector<size_t> &targets)
{
    size_t num_cols = predictions.cols();
    if (targets.size() != num_cols) {
        throw std::invalid_argument("Targets size must match batch size");
    }

    scalar_t total_loss = 0.0f;
    const scalar_t epsilon = 1e-7f;
    for (size_t j = 0; j < num_cols; ++j) {
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

    if (grad_output.rows() != num_classes || grad_output.cols() != batch_size) {
        grad_output.reshape(num_classes, batch_size);
    }

    scalar_t inv_batch_size = 1.0f / static_cast<scalar_t>(batch_size);

#pragma omp parallel for collapse(2) if (num_classes * batch_size > 1000)
    for (size_t i = 0; i < num_classes; ++i) {
        for (size_t b = 0; b < batch_size; ++b) {
            scalar_t target = (i == targets[b] ? 1.0f : 0.0f);
            grad_output(i, b) = (predictions(i, b) - target) * inv_batch_size;
        }
    }
}