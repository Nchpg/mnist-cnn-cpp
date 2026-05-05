#pragma once

#include <cstddef>

#include "layers/layer.hpp"
#include "utils/matrix.hpp"

class BatchNormLayer : public Layer
{
public:
    static constexpr const char *LAYER_NAME = "BATCHNORM";

private:
    size_t channels_;
    size_t spatial_size_;
    Matrix gamma_;
    Matrix beta_;
    Matrix grad_gamma_;
    Matrix grad_beta_;
    Matrix running_mean_;
    Matrix running_var_;
    scalar_t momentum_ = 0.9f;
    scalar_t epsilon_ = 1e-5f;
    bool is_training_ = true;
    Matrix saved_mean_;
    Matrix saved_var_;
    const Matrix *input_ptr_ = nullptr;
    Matrix normalized_;
    Matrix output_;

public:
    BatchNormLayer(size_t channels, size_t spatial_size);
    ~BatchNormLayer() override = default;

    const Matrix &forward(const Matrix &input) override;
    const Matrix &backward(const Matrix &gradient) override;

    void clear_gradients() override;
    void set_training(bool training) override;

    std::vector<Parameter> get_parameters() override;

    void save(std::ostream &os) const override;
    void load(std::istream &is) override;

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D &input_shape) const override;
};