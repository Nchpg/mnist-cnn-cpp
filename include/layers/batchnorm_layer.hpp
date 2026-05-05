#pragma once

#include <vector>

#include "layers/layer.hpp"

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
    Tensor running_mean_;
    Tensor running_var_;
    Tensor saved_mean_;
    Tensor saved_var_;

    const Tensor *input_ptr_ = nullptr;
    Tensor normalized_;
    Tensor output_;
    Tensor grad_input_;

    bool is_training_ = true;

public:
    BatchNormLayer(size_t channels, size_t spatial_size);
    ~BatchNormLayer() override = default;

    const Tensor &forward(const Tensor &input) override;
    const Tensor &backward(const Tensor &gradient) override;

    void clear_gradients() override;

    void set_training(bool training)
    {
        is_training_ = training;
    }

    std::vector<Tensor *> get_weights() override
    {
        return { &gamma_, &beta_ };
    }
    std::vector<Tensor *> get_gradients() override
    {
        return { &grad_gamma_, &grad_beta_ };
    }

    void save(std::ostream &os) const override;
    void load(std::istream &is) override;

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D &input_shape) const override;
};