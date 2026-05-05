#pragma once

#include <vector>

#include "utils/matrix.hpp"

class Loss
{
public:
    virtual ~Loss() = default;
    virtual scalar_t forward(const Matrix &predictions,
                             const std::vector<size_t> &targets) = 0;
    virtual void backward(const Matrix &predictions,
                          const std::vector<size_t> &targets,
                          Matrix &grad_output) = 0;
};