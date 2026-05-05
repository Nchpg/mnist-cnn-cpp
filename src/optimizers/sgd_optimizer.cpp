#include "optimizers/sgd_optimizer.hpp"

SGDOptimizer::SGDOptimizer(scalar_t learning_rate)
    : Optimizer(learning_rate)
{
}

void SGDOptimizer::step()
{
    for (auto &param : parameters_) {
        if (param.value && param.gradient) {
            param.value->subtract_scaled(*param.gradient, learning_rate());
        }
    }
}