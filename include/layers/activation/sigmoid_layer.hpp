#pragma once

#include "layers/layer.hpp"

struct SigmoidContext : public LayerContext
{
    Tensor output;
    Tensor grad_input;
};

class SigmoidLayer : public Layer
{
public:
    static constexpr const char *LAYER_MARKER = "SIGM";

public:
    SigmoidLayer();
    ~SigmoidLayer() override = default;

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
