#pragma once

#include "loss.hpp"

class CrossEntropyLoss : public Loss
{
public:
    scalar_t forward(const Matrix &predictions,
                      const std::vector<size_t> &targets) override;

    void backward(const Matrix &predictions, const std::vector<size_t> &targets,
                 Matrix &grad_output) override;
};