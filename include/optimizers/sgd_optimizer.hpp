#pragma once

#include "optimizers/optimizer.hpp"

class SGDOptimizer : public Optimizer
{
public:
    SGDOptimizer(scalar_t learning_rate);

    void step() override;
};