#pragma once

#include <vector>

#include "layers/layer.hpp"

struct BatchNormContext : public LayerContext
{
    Tensor x_hat;
    Tensor normalized;
    Tensor grad_input;
    Tensor saved_mean;
    Tensor saved_var;
};

class BatchNormLayer : public Layer
{
private:
    size_t channels_;
    size_t spatial_size_;
    scalar_t momentum_ = 0.9f;
    scalar_t epsilon_ = 1e-5f;

    Tensor gamma_;
    Tensor beta_;
    Tensor grad_gamma_;
    Tensor grad_beta_;
    mutable Tensor running_mean_;
    mutable Tensor running_var_;

public:
    BatchNormLayer(size_t channels, size_t spatial_size);
    ~BatchNormLayer() override = default;

    const Tensor& forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const override;
    const Tensor& backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, bool is_training) override;

    void clear_gradients() override;

    std::vector<Tensor*> get_weights() override
    {
        return { &gamma_, &beta_ };
    }
    std::vector<Tensor*> get_gradients() override
    {
        return { &grad_gamma_, &grad_beta_ };
    }

    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D& input_shape) const override;
};