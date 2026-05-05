#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>

#include "layers/layer.hpp"
#include "utils/matrix.hpp"

class DenseLayer : public Layer
{
public:
    static constexpr const char *LAYER_NAME = "DENSE";

private:
    size_t input_size_;
    size_t output_size_;
    Matrix weights_;
    Matrix biases_;
    Matrix activations_;
    const Matrix *input_ptr_ = nullptr;
    Matrix weights_grad_;
    Matrix biases_grad_;

    Matrix weights_t_;
    Matrix input_t_;
    Matrix grad_input_;

public:
    DenseLayer(size_t input_size, size_t output_size, std::mt19937 &gen);
    ~DenseLayer() override = default;

    const Matrix &forward(const Matrix &input) override;
    const Matrix &backward(const Matrix &gradient) override;

    void clear_gradients() override;

    std::vector<Parameter> get_parameters() override;

    const Matrix &activations() const;

    void save(std::ostream &os) const override;
    void load(std::istream &is) override;

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D &input_shape) const override;
};