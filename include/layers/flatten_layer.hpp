#pragma once

#include "layers/layer.hpp"

class FlattenLayer : public Layer
{
private:
    size_t input_channels_;
    size_t input_height_;
    size_t input_width_;

    Tensor output_;
    Tensor grad_input_;

public:
    FlattenLayer(size_t channels, size_t height, size_t width);
    ~FlattenLayer() override = default;

    const Tensor &forward(const Tensor &input) override;
    const Tensor &backward(const Tensor &gradient) override;

    void save(std::ostream &os) const override;
    void load(std::istream &is) override;

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D &input_shape) const override;
};