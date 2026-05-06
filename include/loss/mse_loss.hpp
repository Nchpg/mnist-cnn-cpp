#pragma once

#include "loss.hpp"

class MSELoss : public Loss
{
public:
    scalar_t forward(const Tensor& predictions, const std::vector<size_t>& targets) override;

    void backward(const Tensor& predictions, const std::vector<size_t>& targets, Tensor& grad_output) override;

    bool supports_fusion_with(const std::string& layer_type) const override
    {
        (void)layer_type;
        return false;
    }
    void backward_fused(const std::string& layer_type, const Tensor& probs, const std::vector<size_t>& targets,
                        Tensor& grad_logits) override
    {
        (void)layer_type;
        (void)probs;
        (void)targets;
        (void)grad_logits;
        throw std::runtime_error("Fusion not supported for MSELoss");
    }
};