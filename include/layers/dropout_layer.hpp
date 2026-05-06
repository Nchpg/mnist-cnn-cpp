#pragma once

#include <random>
#include <vector>

#include "layers/layer.hpp"

struct DropoutContext : public LayerContext
{
    Tensor mask;
    Tensor output;
    Tensor grad_input;
};

class DropoutLayer : public Layer
{
private:
    scalar_t ratio_;
    std::mt19937 &local_gen_;

public:
    DropoutLayer(scalar_t ratio, std::mt19937 &gen);
    ~DropoutLayer() override = default;

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