#include "layers/activation/activation.hpp"

#include <cmath>
#include <omp.h>
#include <stdexcept>
namespace Activation
{
/*
 * RELU: f(x) = max(0, x)
 *
 * BACKPROPAGATION (CHAIN RULE):
 * -----------------------------
 * Goal: Compute dL/dx = dL/dy * dy/dx
 *
 * 1. Local Gradient:
 *    dy/dx = (x > 0) ? 1 : 0
 *
 * 2. Applying Chain Rule:
 *    dL/dx = dL/dy * (1 if x > 0 else 0)
 *
 * 3. Final Implementation:
 *    dL/dx = (x > 0) ? dL/dy : 0
 */
void relu(const Tensor& in, Tensor& out)
{
    out = in;
    out.map([](scalar_t x) { return x > 0.0f ? x : 0.0f; });
}

void relu_backward(const Tensor& in, const Tensor& grad_out, Tensor& grad_in)
{
    Tensor::elementwise(in, grad_out, grad_in, [](scalar_t x, scalar_t g) { return x > 0.0f ? g : 0.0f; });
}

/*
 * SIGMOID: f(x) = 1 / (1 + exp(-x))
 *
 * BACKPROPAGATION (CHAIN RULE):
 * -----------------------------
 * Goal: Compute dL/dx = dL/dy * dy/dx
 *
 * 1. Local Gradient:
 *    dy/dx = y * (1 - y)
 *
 * 2. Applying Chain Rule:
 *    dL/dx = dL/dy * dy/dx
 *          = dL/dy * [ y * (1 - y) ]
 *
 * 3. Final Implementation:
 *    dL/dx = dL/dy * y * (1 - y)
 */
void sigmoid(const Tensor& in, Tensor& out)
{
    out = in;
    out.map([](scalar_t x) { return 1.0f / (1.0f + std::exp(-x)); });
}

void sigmoid_backward(const Tensor& out, const Tensor& grad_out, Tensor& grad_in)
{
    Tensor::elementwise(grad_out, out, grad_in, [](scalar_t g, scalar_t o) { return g * o * (1.0f - o); });
}

/*
 * SOFTMAX: y_i = exp(x_i - max(X)) / Σ_k exp(x_k - max(X))
 *
 * BACKPROPAGATION (CHAIN RULE):
 * -----------------------------
 * Goal: Compute dL/dx_j = Σ_i (dL/dy_i * dy_i/dx_j)
 *
 * 1. Local Gradient (Jacobian):
 *    dy_i / dx_j = y_i * (δ_ij - y_j)
 *    where δ_ij = 1 if i == j, else 0.
 *
 * 2. Applying Chain Rule:
 *    dL/dx_j = (dL/dy_j * dy_j/dx_j) + Σ_{i≠j} (dL/dy_i * dy_i/dx_j)
 *            = dL/dy_j * y_j(1 - y_j) + Σ_{i≠j} (dL/dy_i * (-y_i * y_j))
 *
 * 3. Final Simplified Form:
 *    dL/dx_j = y_j * [ dL/dy_j - Σ_i (dL/dy_i * y_i) ]
 */
void softmax(const Tensor& in, Tensor& out)
{
    const Shape input_shape = in.shape();
    const size_t batch_size = input_shape.batch();
    const size_t num_classes = input_shape.classes();
    out.reshape(input_shape);

#pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b)
    {
        // 1. Find max for numerical stability
        scalar_t max_val = in(b, 0);
        for (size_t c = 1; c < num_classes; ++c)
            if (in(b, c) > max_val) max_val = in(b, c);

        // 2. Compute exp(x - max_val)
        for (size_t c = 0; c < num_classes; ++c)
            out(b, c) = std::exp(in(b, c) - max_val);

        // 3. Compute sum and normalize
        scalar_t sum = 0.0f;
        for (size_t c = 0; c < num_classes; ++c)
            sum += out(b, c);

        const scalar_t inv_sum = 1.0f / sum;
        for (size_t c = 0; c < num_classes; ++c)
            out(b, c) *= inv_sum;
    }
}

void softmax_backward(const Tensor& out, const Tensor& grad_out, Tensor& grad_in)
{
    const Shape output_shape = out.shape();
    const size_t batch_size = output_shape.batch();
    const size_t num_classes = output_shape.classes();
    grad_in.reshape(output_shape);

#pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b)
    {
        // Calculate dot product: sum(out * grad_out)
        scalar_t dot = 0.0f;
        for (size_t k = 0; k < num_classes; ++k)
            dot += out(b, k) * grad_out(b, k);

        // grad_in = out * (grad_out - dot)
        for (size_t k = 0; k < num_classes; ++k)
            grad_in(b, k) = out(b, k) * (grad_out(b, k) - dot);
    }
}
} // namespace Activation