#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "utils/tensor.hpp"

class Loss
{
public:
    virtual ~Loss() = default;
    virtual scalar_t forward(const Tensor &predictions,
                             const std::vector<size_t> &targets) = 0;
    virtual void backward(const Tensor &predictions,
                          const std::vector<size_t> &targets,
                          Tensor &grad_output) = 0;

    virtual bool supports_fusion_with(const std::string &layer_type) const
    {
        (void)layer_type;
        return false;
    }

    virtual void backward_fused(const std::string &layer_type,
                                const Tensor &probs,
                                const std::vector<size_t> &targets,
                                Tensor &grad_logits)
    {
        (void)layer_type;
        (void)probs;
        (void)targets;
        (void)grad_logits;
        throw std::runtime_error("Fusion not supported for this layer type");
    }
};