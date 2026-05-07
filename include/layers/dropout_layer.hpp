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

public:
    explicit DropoutLayer(scalar_t ratio);
    ~DropoutLayer() override = default;

    const Tensor& forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const override;
    const Tensor& backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, bool is_training) override;

    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    nlohmann::json get_config() const override;

    Shape get_output_shape(const Shape& input_shape) const override;
    Shape get_input_shape(const Shape& output_shape) const override;
};