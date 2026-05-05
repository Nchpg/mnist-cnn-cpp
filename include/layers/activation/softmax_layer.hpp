#pragma once

#include "layers/activation/activation.hpp"
#include "layers/layer.hpp"

class SoftmaxLayer : public Layer
{
public:
    static constexpr const char *LAYER_MARKER = "SOFT";

private:
    Tensor output_;
    Tensor grad_input_;

public:
    SoftmaxLayer();
    ~SoftmaxLayer() override;

    const Tensor &forward(const Tensor &input) override;
    const Tensor &backward(const Tensor &gradient) override;

    nlohmann::json get_config() const override;
    Shape3D get_output_shape(const Shape3D &input_shape) const override;

    void save(std::ostream &os) const override;
    void load(std::istream &is) override;
};