#pragma once

#include <random>
#include <stdexcept>
#include <string>

#include "layers/layer.hpp"

class DropoutLayer : public Layer
{
public:
    static constexpr const char *LAYER_MARKER = "DROP";

private:
    scalar_t ratio_;
    bool is_training_ = true;
    Matrix mask_;
    Matrix output_;
    Matrix grad_input_;
    inline static thread_local std::mt19937 local_gen_;

public:
    DropoutLayer(scalar_t ratio, std::mt19937 &gen);

    void set_training(bool training) override;

    const Matrix &forward(const Matrix &input) override;

    const Matrix &backward(const Matrix &gradient) override;

    Shape3D get_output_shape(const Shape3D &input_shape) const override;

    void save(std::ostream &os) const override;
    void load(std::istream &is) override;

    nlohmann::json get_config() const override;
};
