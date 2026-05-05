#pragma once

#include "layers/activation/activation.hpp"
#include "layers/layer.hpp"
#include "utils/matrix.hpp"

class SoftmaxLayer : public Layer
{
private:
    Matrix output_;
    Matrix grad_input_;

public:
    SoftmaxLayer();
    ~SoftmaxLayer() override;

    const Matrix &forward(const Matrix &input) override;
    const Matrix &backward(const Matrix &gradient) override;

    nlohmann::json get_config() const override;
    Shape3D get_output_shape(const Shape3D &input_shape) const override;
};