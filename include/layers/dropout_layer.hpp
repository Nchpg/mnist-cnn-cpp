#pragma once

#include <random>
#include <vector>

#include "layers/layer.hpp"

class DropoutLayer : public Layer
{
private:
    scalar_t ratio_;
    std::mt19937 &local_gen_;

    Tensor mask_;
    Tensor output_;
    Tensor grad_input_;
    bool is_training_ = true;

public:
    DropoutLayer(scalar_t ratio, std::mt19937 &gen);
    ~DropoutLayer() override = default;

    const Tensor &forward(const Tensor &input) override;
    const Tensor &backward(const Tensor &gradient) override;

    void save(std::ostream &os) const override;
    void load(std::istream &is) override;

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D &input_shape) const override;
};