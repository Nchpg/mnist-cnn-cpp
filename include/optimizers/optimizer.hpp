#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "layers/layer.hpp"

class Optimizer
{
private:
    scalar_t learning_rate_;

protected:
    std::vector<Parameter> parameters_;

public:
    Optimizer(scalar_t learning_rate = 0.0f)
        : learning_rate_(learning_rate)
    {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    virtual void reset()
    {}
    virtual void set_learning_rate(scalar_t lr)
    {
        learning_rate_ = lr;
    }
    scalar_t learning_rate() const
    {
        return learning_rate_;
    }

    virtual void add_parameters(const std::vector<Parameter> &params)
    {
        parameters_.insert(parameters_.end(), params.begin(), params.end());
    }
};