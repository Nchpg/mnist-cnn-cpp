#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "utils/tensor.hpp"

class Optimizer
{
private:
    scalar_t learning_rate_;

protected:
    std::vector<Tensor*> weights_;
    std::vector<Tensor*> gradients_;

public:
    explicit Optimizer(scalar_t learning_rate = 0.0f) noexcept
        : learning_rate_(learning_rate)
    {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    virtual void reset()
    {}
    virtual void set_learning_rate(scalar_t lr) noexcept
    {
        learning_rate_ = lr;
    }
    scalar_t learning_rate() const noexcept
    {
        return learning_rate_;
    }

    virtual void set_parameters(const std::vector<Tensor*>& weights, const std::vector<Tensor*>& grads)
    {
        weights_ = weights;
        gradients_ = grads;
    }
};