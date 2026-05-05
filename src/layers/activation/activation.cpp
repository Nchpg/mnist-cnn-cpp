#include "layers/activation/activation.hpp"

#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace Activation
{
    // --- ReLU ---
    void relu(const Matrix &in, Matrix &out)
    {
        if (out.rows() != in.rows() || out.cols() != in.cols())
            out.reshape(in.rows(), in.cols());

        size_t n = in.size();
        const scalar_t *in_ptr = in.data();
        scalar_t *out_ptr = out.data();

#pragma omp parallel for
        for (size_t i = 0; i < n; ++i)
        {
            out_ptr[i] = in_ptr[i] > 0.0f ? in_ptr[i] : 0.0f;
        }
    }

    void relu_backward(const Matrix &in, const Matrix &grad_out, Matrix &grad_in)
    {
        if (grad_in.rows() != in.rows() || grad_in.cols() != in.cols())
            grad_in.reshape(in.rows(), in.cols());

        size_t n = in.size();
        const scalar_t *in_ptr = in.data();
        const scalar_t *grad_out_ptr = grad_out.data();
        scalar_t *grad_in_ptr = grad_in.data();

#pragma omp parallel for
        for (size_t i = 0; i < n; ++i)
        {
            grad_in_ptr[i] = in_ptr[i] > 0.0f ? grad_out_ptr[i] : 0.0f;
        }
    }

    // --- Sigmoid ---
    void sigmoid(const Matrix &in, Matrix &out)
    {
        if (out.rows() != in.rows() || out.cols() != in.cols())
            out.reshape(in.rows(), in.cols());

        size_t n = in.size();
        const scalar_t *in_ptr = in.data();
        scalar_t *out_ptr = out.data();

#pragma omp parallel for
        for (size_t i = 0; i < n; ++i)
        {
            out_ptr[i] = 1.0f / (1.0f + std::exp(-in_ptr[i]));
        }
    }

    void sigmoid_backward(const Matrix &out, const Matrix &grad_out,
                          Matrix &grad_in)
    {
        if (grad_in.rows() != out.rows() || grad_in.cols() != out.cols())
            grad_in.reshape(out.rows(), out.cols());

        size_t n = out.size();
        const scalar_t *out_ptr = out.data();
        const scalar_t *grad_out_ptr = grad_out.data();
        scalar_t *grad_in_ptr = grad_in.data();

#pragma omp parallel for
        for (size_t i = 0; i < n; ++i)
        {
            grad_in_ptr[i] = grad_out_ptr[i] * out_ptr[i] * (1.0f - out_ptr[i]);
        }
    }

    // --- Softmax ---
    void softmax(const Matrix &in, Matrix &out)
    {
        size_t num_rows = in.rows();
        size_t cols = in.cols();
        if (out.rows() != num_rows || out.cols() != cols)
        {
            out.reshape(num_rows, cols);
        }

#pragma omp parallel for
        for (size_t j = 0; j < cols; ++j)
        {
            scalar_t max_val = in(0, j);
            for (size_t i = 1; i < num_rows; ++i)
            {
                if (in(i, j) > max_val)
                    max_val = in(i, j);
            }

            scalar_t sum = 0.0f;
            for (size_t i = 0; i < num_rows; ++i)
            {
                out(i, j) = std::exp(in(i, j) - max_val);
                sum += out(i, j);
            }
            scalar_t inv_sum = 1.0f / sum;
            for (size_t i = 0; i < num_rows; ++i)
            {
                out(i, j) *= inv_sum;
            }
        }
    }

    void softmax_backward(const Matrix &out, const Matrix &grad_out,
                          Matrix &grad_in)
    {
        size_t num_classes = out.rows();
        size_t batch_size = out.cols();

        if (grad_in.rows() != num_classes || grad_in.cols() != batch_size)
        {
            grad_in.reshape(num_classes, batch_size);
        }

#pragma omp parallel for
        for (size_t b = 0; b < batch_size; ++b)
        {
            scalar_t dot = 0.0f;
            for (size_t k = 0; k < num_classes; ++k)
            {
                dot += out(k, b) * grad_out(k, b);
            }

            for (size_t k = 0; k < num_classes; ++k)
            {
                grad_in(k, b) = out(k, b) * (grad_out(k, b) - dot);
            }
        }
    }
} // namespace Activation
