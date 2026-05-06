#pragma once

#include <algorithm>

#include "layers/layer.hpp"

struct ReluContext : public LayerContext
{
    Tensor input;
    Tensor output;
    Tensor grad_input;
};

class ReluLayer : public Layer
{
public:
    static constexpr const char *LAYER_MARKER = "RELU";

public:
    ReluLayer();
    ~ReluLayer() override = default;

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