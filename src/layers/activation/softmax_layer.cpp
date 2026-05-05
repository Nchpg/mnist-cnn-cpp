#include "layers/activation/softmax_layer.hpp"

namespace
{

    void softmax_backward(const Matrix &output, const Matrix &gradient,
                          Matrix &grad_input)
    {
        size_t num_classes = output.rows();
        size_t batch_size = output.cols();

        std::vector<scalar_t> sums(batch_size, 0.0f);

        for (size_t k = 0; k < num_classes; ++k)
        {
            for (size_t b = 0; b < batch_size; ++b)
            {
                sums[b] += output(k, b) * gradient(k, b);
            }
        }

        for (size_t k = 0; k < num_classes; ++k)
        {
            for (size_t b = 0; b < batch_size; ++b)
            {
                grad_input(k, b) = output(k, b) * (gradient(k, b) - sums[b]);
            }
        }
    }
} // namespace

SoftmaxLayer::SoftmaxLayer()
{}

SoftmaxLayer::~SoftmaxLayer()
{}

const Matrix &SoftmaxLayer::forward(const Matrix &input)
{
    output_.reshape(input.rows(), input.cols());
    Activation::softmax(input, output_);
    return output_;
}

const Matrix &SoftmaxLayer::backward(const Matrix &gradient)
{
    grad_input_.reshape(gradient.rows(), gradient.cols());
    softmax_backward(output_, gradient, grad_input_);
    return grad_input_;
}

nlohmann::json SoftmaxLayer::get_config() const
{
    return { { "type", "Softmax" } };
}

Shape3D SoftmaxLayer::get_output_shape(const Shape3D &input_shape) const
{
    return input_shape;
}