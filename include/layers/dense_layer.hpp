#pragma once

#include <random>
#include <vector>

#include "layers/layer.hpp"

struct DenseContext : public LayerContext
{
    const Tensor* input_ptr = nullptr;
    Tensor activations;
    Tensor grad_input;
};

class DenseLayer : public Layer
{
private:
    size_t input_size_;
    size_t output_size_;

    Tensor weights_;
    Tensor biases_;

    Tensor weights_grad_;
    Tensor biases_grad_;

public:
    DenseLayer(size_t input_size, size_t output_size, std::mt19937& gen);
    ~DenseLayer() override = default;

    const Tensor& forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const override;
    const Tensor& backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, bool is_training) override;

    void clear_gradients() override;

    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    std::vector<Tensor*> get_weights() override
    {
        return { &weights_, &biases_ };
    }
    std::vector<Tensor*> get_gradients() override
    {
        return { &weights_grad_, &biases_grad_ };
    }

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D& input_shape) const override;
};