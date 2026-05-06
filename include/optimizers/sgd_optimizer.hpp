#pragma once

#include "optimizers/optimizer.hpp"

class SGDOptimizer : public Optimizer
{
public:
    explicit SGDOptimizer(scalar_t learning_rate);

    void step() override;
};