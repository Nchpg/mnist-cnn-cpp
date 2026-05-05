#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "utils/tensor.hpp"
#include "utils/utils.hpp"

namespace Activation
{
    void relu(const Tensor &in, Tensor &out);
    void relu_backward(const Tensor &in, const Tensor &grad_out,
                       Tensor &grad_in);

    void sigmoid(const Tensor &in, Tensor &out);
    void sigmoid_backward(const Tensor &out, const Tensor &grad_out,
                          Tensor &grad_in);

    void softmax(const Tensor &in, Tensor &out);
    void softmax_backward(const Tensor &out, const Tensor &grad_out,
                          Tensor &grad_in);
} // namespace Activation