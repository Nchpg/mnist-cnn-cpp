#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "utils/matrix.hpp"
#include "utils/utils.hpp"

namespace Activation
{
    scalar_t relu(scalar_t x);
    scalar_t relu_deriv(scalar_t x);

    void softmax(const Matrix &logits, Matrix &probs);
} // namespace Activation