#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "utils/matrix.hpp"
#include "utils/utils.hpp"

namespace Activation
{
    // ReLU
    void relu(const Matrix &in, Matrix &out);
    void relu_backward(const Matrix &in, const Matrix &grad_out, Matrix &grad_in);

    // Sigmoid
    void sigmoid(const Matrix &in, Matrix &out);
    void sigmoid_backward(const Matrix &out, const Matrix &grad_out,
                          Matrix &grad_in);

    // Softmax
    void softmax(const Matrix &in, Matrix &out);
    void softmax_backward(const Matrix &out, const Matrix &grad_out,
                          Matrix &grad_in);
} // namespace Activation