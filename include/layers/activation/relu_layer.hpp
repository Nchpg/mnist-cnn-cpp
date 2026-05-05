#pragma once

#include <algorithm>

#include "layers/layer.hpp"
#include "utils/matrix.hpp"

class ReluLayer : public Layer
{
public:
    static constexpr const char *LAYER_MARKER = "RELU";

private:
    const Matrix *input_ptr_ = nullptr;
    Matrix output_;
    Matrix grad_input_;

public:
    ReluLayer();
    ~ReluLayer() override = default;

    const Matrix &forward(const Matrix &input) override;

    const Matrix &backward(const Matrix &gradient) override;

    void save(std::ostream &os) const override;
    void load(std::istream &is) override;

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D &input_shape) const override;
};