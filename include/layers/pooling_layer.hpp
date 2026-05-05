#pragma once

#include <random>
#include <vector>

#include "layers/layer.hpp"

class PoolingLayer : public Layer
{
private:
    size_t in_h_, in_w_, in_c_;
    size_t pool_size_;
    size_t stride_;
    size_t out_h_, out_w_;

    const Tensor *input_ptr_ = nullptr;
    Tensor output_;
    Tensor grad_input_;
    std::vector<size_t> argmax_indices_;

public:
    PoolingLayer(size_t input_h, size_t input_w, size_t input_c,
                 size_t pool_size, size_t stride = 2);
    ~PoolingLayer() override = default;

    const Tensor &forward(const Tensor &input) override;
    const Tensor &backward(const Tensor &gradient) override;

    void save(std::ostream &os) const override;
    void load(std::istream &is) override;

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D &input_shape) const override;
};