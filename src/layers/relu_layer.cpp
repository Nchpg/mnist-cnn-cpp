#include "layers/relu_layer.hpp"

#include <omp.h>

ReluLayer::ReluLayer()
{}

const Matrix &ReluLayer::forward(const Matrix &input)
{
    input_ptr_ = &input;
    if (output_.rows() != input.rows() || output_.cols() != input.cols())
    {
        output_.reshape(input.rows(), input.cols());
    }

    size_t n = input.size();
    const scalar_t *in_ptr = input.data();
    scalar_t *out_ptr = output_.data();

#pragma omp simd
    for (size_t i = 0; i < n; ++i)
    {
        out_ptr[i] = in_ptr[i] > 0.0f ? in_ptr[i] : 0.0f;
    }
    return output_;
}

const Matrix &ReluLayer::backward(const Matrix &gradient)
{
    if (grad_input_.rows() != input_ptr_->rows()
        || grad_input_.cols() != input_ptr_->cols())
    {
        grad_input_.reshape(input_ptr_->rows(), input_ptr_->cols());
    }

    size_t n = input_ptr_->size();
    const scalar_t *in_ptr = input_ptr_->data();
    const scalar_t *grad_ptr = gradient.data();
    scalar_t *out_grad_ptr = grad_input_.data();

#pragma omp simd
    for (size_t i = 0; i < n; ++i)
    {
        out_grad_ptr[i] = in_ptr[i] > 0.0f ? grad_ptr[i] : 0.0f;
    }
    return grad_input_;
}

void ReluLayer::save(std::ostream &os) const
{
    uint32_t marker = make_marker(LAYER_MARKER);
    os.write(reinterpret_cast<const char *>(&marker), sizeof(marker));
}

void ReluLayer::load(std::istream &is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    if (marker != make_marker(LAYER_MARKER))
        throw std::runtime_error("Architecture mismatch in ReluLayer load");
}

nlohmann::json ReluLayer::get_config() const
{
    return { { "type", "ReLU" } };
}

Shape3D ReluLayer::get_output_shape(const Shape3D &input_shape) const
{
    return input_shape;
}