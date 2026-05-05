#pragma once

#include <vector>

#include "layers/layer.hpp"
#include "utils/matrix.hpp"
#include "utils/tensor_batch.hpp"

class PoolingLayer : public Layer
{
public:
    static constexpr const char *LAYER_MARKER = "POOL";

private:
    size_t pool_size_;
    size_t stride_;

    const Matrix *input_ptr_ = nullptr;
    TensorBatch output_;
    TensorBatch grad_input_;

    std::vector<size_t> argmax_indices_;

public:
    PoolingLayer(size_t input_rows, size_t input_cols, size_t filter_count,
                 size_t pool_size, size_t stride);
    ~PoolingLayer() override = default;

    const Matrix &forward(const Matrix &input) override;
    const Matrix &backward(const Matrix &gradient) override;

    void save(std::ostream &os) const override;
    void load(std::istream &is) override;

    nlohmann::json get_config() const override;

    Shape3D get_output_shape(const Shape3D &input_shape) const override;
};