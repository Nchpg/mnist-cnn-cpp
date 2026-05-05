#pragma once

#include <algorithm>

#include "layers/layer.hpp"

class ReluLayer : public Layer
{
public:
    static constexpr const char *LAYER_MARKER = "RELU";

private:
    const Tensor *input_ptr_ = nullptr;
    Tensor output_;
    Tensor grad_input_;

public:
    ReluLayer();
    ~ReluLayer() override = default;

    const Tensor &forward(const Tensor &input) override;

    const Tensor &backward(const Tensor &gradient) override;

    void save(std::ostream &os) const override;
    void load(std::istream &is) override;

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D &input_shape) const override;
};