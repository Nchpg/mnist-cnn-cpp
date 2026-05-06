#pragma once

#include "layers/activation/activation.hpp"
#include "layers/layer.hpp"

struct SoftmaxContext : public LayerContext
{
    Tensor output;
    Tensor grad_input;
};

class SoftmaxLayer : public Layer
{
public:
    static constexpr const char* LAYER_MARKER = "SOFT";

public:
    SoftmaxLayer();
    ~SoftmaxLayer() override;

    const Tensor& forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const override;
    const Tensor& backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, bool is_training) override;

    nlohmann::json get_config() const override;
    Shape3D get_output_shape(const Shape3D& input_shape) const override;

    void save(std::ostream& os) const override;
    void load(std::istream& is) override;
};