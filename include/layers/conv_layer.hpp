#pragma once

#include <random>
#include <vector>

#include "layers/layer.hpp"

struct ConvContext : public LayerContext
{
    Tensor output;
    Tensor grad_input;
    Tensor col_buffer;
    Tensor grad_view;
    Tensor grad_col;
    Tensor gemm_out;
};

class ConvLayer : public Layer
{
private:
    size_t in_h_, in_w_, in_c_;
    size_t k_size_;
    size_t out_c_;
    size_t out_h_, out_w_;

    Tensor filters_;
    Tensor biases_;

    Tensor filters_grad_;
    Tensor biases_grad_;

public:
    ConvLayer(size_t input_h, size_t input_w, size_t input_c, size_t kernel_size, size_t filter_count,
              std::mt19937& gen);
    ~ConvLayer() override = default;

    const Tensor& forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const override;
    const Tensor& backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, bool is_training) override;

    void clear_gradients() override;

    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    std::vector<Tensor*> get_weights() override
    {
        return { &filters_, &biases_ };
    }
    std::vector<Tensor*> get_gradients() override
    {
        return { &filters_grad_, &biases_grad_ };
    }

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D& input_shape) const override;
};