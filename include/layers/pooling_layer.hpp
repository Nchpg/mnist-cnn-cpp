#pragma once

#include <random>
#include <vector>

#include "layers/layer.hpp"

struct PoolingContext : public LayerContext
{
    Tensor output;
    Tensor grad_input;
    std::vector<size_t> argmax_indices;
};

class PoolingLayer : public Layer
{
private:
    size_t in_h_, in_w_, in_c_;
    size_t pool_size_;
    size_t stride_;
    size_t out_h_, out_w_;

public:
    PoolingLayer(size_t input_h, size_t input_w, size_t input_c,
                 size_t pool_size, size_t stride = 2);
    ~PoolingLayer() override = default;

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