#include "layers/activation/activation.hpp"

#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace Activation
{
void relu(const Tensor& in, Tensor& out)
{
    if (out.shape() != in.shape())
        out.reshape(in.shape());

    const size_t n = in.size();
    const scalar_t* in_ptr = in.data_ptr();
    scalar_t* out_ptr = out.data_ptr();

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
    {
        out_ptr[i] = in_ptr[i] > 0.0f ? in_ptr[i] : 0.0f;
    }
}

void relu_backward(const Tensor& in, const Tensor& grad_out, Tensor& grad_in)
{
    if (grad_in.shape() != in.shape())
        grad_in.reshape(in.shape());

    const size_t n = in.size();
    const scalar_t* in_ptr = in.data_ptr();
    const scalar_t* grad_out_ptr = grad_out.data_ptr();
    scalar_t* grad_in_ptr = grad_in.data_ptr();

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
    {
        grad_in_ptr[i] = in_ptr[i] > 0.0f ? grad_out_ptr[i] : 0.0f;
    }
}

void sigmoid(const Tensor& in, Tensor& out)
{
    if (out.shape() != in.shape())
        out.reshape(in.shape());

    const size_t n = in.size();
    const scalar_t* in_ptr = in.data_ptr();
    scalar_t* out_ptr = out.data_ptr();

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
    {
        out_ptr[i] = 1.0f / (1.0f + std::exp(-in_ptr[i]));
    }
}

void sigmoid_backward(const Tensor& out, const Tensor& grad_out, Tensor& grad_in)
{
    if (grad_in.shape() != out.shape())
        grad_in.reshape(out.shape());

    const size_t n = out.size();
    const scalar_t* out_ptr = out.data_ptr();
    const scalar_t* grad_out_ptr = grad_out.data_ptr();
    scalar_t* grad_in_ptr = grad_in.data_ptr();

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
    {
        grad_in_ptr[i] = grad_out_ptr[i] * out_ptr[i] * (1.0f - out_ptr[i]);
    }
}

void softmax(const Tensor& in, Tensor& out)
{
    const size_t batch_size = in.shape()[0];
    const size_t num_classes = in.shape()[1];
    if (out.rank() != 2 || out.shape()[0] != batch_size || out.shape()[1] != num_classes)
    {
        out.reshape(Shape({ batch_size, num_classes }));
    }

#pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b)
    {
        scalar_t max_val = in(b, 0);
        for (size_t c = 1; c < num_classes; ++c)
        {
            if (in(b, c) > max_val)
                max_val = in(b, c);
        }

        scalar_t sum = 0.0f;
        for (size_t c = 0; c < num_classes; ++c)
        {
            out(b, c) = std::exp(in(b, c) - max_val);
            sum += out(b, c);
        }
        const scalar_t inv_sum = 1.0f / sum;
        for (size_t c = 0; c < num_classes; ++c)
        {
            out(b, c) *= inv_sum;
        }
    }
}

void softmax_backward(const Tensor& out, const Tensor& grad_out, Tensor& grad_in)
{
    const size_t batch_size = out.shape()[0];
    const size_t num_classes = out.shape()[1];

    if (grad_in.rank() != 2 || grad_in.shape()[0] != batch_size || grad_in.shape()[1] != num_classes)
    {
        grad_in.reshape(Shape({ batch_size, num_classes }));
    }

#pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b)
    {
        scalar_t dot = 0.0f;
        for (size_t k = 0; k < num_classes; ++k)
        {
            dot += out(b, k) * grad_out(b, k);
        }

        for (size_t k = 0; k < num_classes; ++k)
        {
            grad_in(b, k) = out(b, k) * (grad_out(b, k) - dot);
        }
    }
}
} // namespace Activation