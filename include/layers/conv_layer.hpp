#pragma once

#include <random>
#include <vector>

#include "layers/layer.hpp"
#include "utils/matrix.hpp"
#include "utils/tensor_batch.hpp"

class ConvLayer : public Layer
{
public:
    static constexpr const char *LAYER_NAME = "CONV";

private:
    size_t kernel_size_;
    Matrix biases_;
    Matrix biases_grad_;

    const Matrix *input_ptr_ = nullptr;
    TensorBatch output_;
    TensorBatch grad_input_;

    Matrix filters_matrix_;
    Matrix filters_grad_matrix_;

    Matrix im2col_buffer_;
    Matrix gemm_out_;
    Matrix grad_col_buffer_;

    void im2col(const Matrix &input, Matrix &out_col);
    void col2im(const Matrix &grad_col, TensorBatch &grad_input);

public:
    ConvLayer(size_t input_rows, size_t input_cols, size_t input_channels,
              size_t kernel_size, size_t filter_count, std::mt19937 &gen,
              scalar_t weight_scale = 0.01f);
    ~ConvLayer() override = default;

    const Matrix &forward(const Matrix &input) override;
    const Matrix &backward(const Matrix &gradient) override;

    void clear_gradients() override;

    std::vector<Parameter> get_parameters() override;

    void save(std::ostream &os) const override;
    void load(std::istream &is) override;

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D &input_shape) const override;
};
