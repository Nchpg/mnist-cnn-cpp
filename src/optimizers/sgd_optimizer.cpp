#include "optimizers/sgd_optimizer.hpp"

SGDOptimizer::SGDOptimizer(scalar_t learning_rate)
    : Optimizer(learning_rate)
{}

void SGDOptimizer::step()
{
    for (size_t i = 0; i < weights_.size(); ++i)
    {
        if (weights_[i] && gradients_[i])
        {
            weights_[i]->add_scaled(*gradients_[i], -learning_rate());
        }
    }
}