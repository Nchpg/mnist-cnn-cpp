#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>

#include "layers/layer.hpp"

class FlattenLayer : public Layer
{
public:
    static constexpr const char *LAYER_MARKER = "FLAT";

public:
    FlattenLayer();
    ~FlattenLayer() override = default;

    const Matrix &forward(const Matrix &input) override;

    const Matrix &backward(const Matrix &gradient) override;

    void save(std::ostream &os) const override;

    void load(std::istream &is) override;

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D &input_shape) const override;
};