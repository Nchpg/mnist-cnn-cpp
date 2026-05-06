#pragma once

#include "optimizers/optimizer.hpp"

class AdamOptimizer : public Optimizer
{
private:
    scalar_t beta1_;
    scalar_t beta2_;
    scalar_t epsilon_;
    size_t t_;
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;

public:
    AdamOptimizer(scalar_t learning_rate, scalar_t beta1 = 0.9f, scalar_t beta2 = 0.999f, scalar_t epsilon = 1e-8f);

    void set_parameters(const std::vector<Tensor*>& weights, const std::vector<Tensor*>& grads) override;

    void step() override;

    void reset() override;
};