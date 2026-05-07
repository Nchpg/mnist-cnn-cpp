#pragma once

#include "core/model.hpp"
#include "utils/tensor.hpp"

enum class OptimizerType
{
    SGD,
    Adam
};

enum class LossType
{
    CrossEntropy,
    MSE
};

struct Hyperparameters
{
    size_t batch_size = 64;
    OptimizerType optimizer_type = OptimizerType::Adam;
    LossType loss_type = LossType::CrossEntropy;
    scalar_t learning_rate = 0.001f;
    scalar_t beta1 = 0.9f;
    scalar_t beta2 = 0.999f;
    scalar_t epsilon = 1e-8f;
};