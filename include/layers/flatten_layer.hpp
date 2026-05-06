#pragma once

#include "layers/layer.hpp"

struct FlattenContext : public LayerContext
{
    Tensor output;
    Tensor grad_input;
};

class FlattenLayer : public Layer
{
private:
    size_t input_channels_;
    size_t input_height_;
    size_t input_width_;

public:
    FlattenLayer(size_t channels, size_t height, size_t width);
    ~FlattenLayer() override = default;

    const Tensor &forward(const Tensor &input,
                          std::unique_ptr<LayerContext> &ctx,
                          bool is_training) const override;
    const Tensor &backward(const Tensor &gradient,
                           std::unique_ptr<LayerContext> &ctx,
                           bool is_training) override;

    void save(std::ostream &os) const override;
    void load(std::istream &is) override;

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D &input_shape) const override;
};